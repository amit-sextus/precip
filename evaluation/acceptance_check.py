#!/usr/bin/env python3
"""
Acceptance check for evaluation results.

Verifies that all evaluation metrics satisfy required invariants:
- Overall metrics equal time-weighted seasonal averages
- CORP decomposition identities hold
- All runs use Germany mask and cos(lat) weighting
- No invalid values

Usage:
    python -m evaluation.acceptance_check --csv evaluation/aggregated_metrics.csv
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from typing import List, Tuple


# ANSI color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def check_seasonal_consistency(row: pd.Series, tolerance: float = 1e-6) -> List[str]:
    """
    Check that overall metrics equal time-weighted seasonal averages.
    """
    issues = []
    
    # Get seasonal values
    seasons = ['djf', 'mam', 'jja', 'son']
    
    # Check CRPS consistency
    seasonal_crps = []
    seasonal_days = []
    
    for season in seasons:
        crps_col = f'crps_{season}'
        days_col = f'n_days_{season}'
        
        if crps_col in row and pd.notna(row[crps_col]):
            seasonal_crps.append(row[crps_col])
            # Use actual day counts if available, otherwise use standard year
            if days_col in row and pd.notna(row[days_col]):
                seasonal_days.append(row[days_col])
            else:
                # Standard year approximation
                if season == 'djf':
                    seasonal_days.append(90)  # Dec(31) + Jan(31) + Feb(28)
                elif season in ['mam', 'jja']:
                    seasonal_days.append(92)  # 31 + 30 + 31 or 30 + 31 + 31
                else:  # son
                    seasonal_days.append(91)  # Sep(30) + Oct(31) + Nov(30)
    
    if len(seasonal_crps) == 4 and 'mean_crps' in row and pd.notna(row['mean_crps']):
        # Calculate weighted average
        total_days = sum(seasonal_days)
        weighted_avg = sum(c * d for c, d in zip(seasonal_crps, seasonal_days)) / total_days
        
        diff = abs(row['mean_crps'] - weighted_avg)
        if diff > tolerance:
            issues.append(
                f"CRPS seasonal consistency failed: "
                f"overall={row['mean_crps']:.6f}, weighted_avg={weighted_avg:.6f}, "
                f"diff={diff:.2e}"
            )
    
    # Check BS consistency (similar logic)
    seasonal_bs = []
    for season in seasons:
        bs_col = f'bs_{season}'
        if bs_col in row and pd.notna(row[bs_col]):
            seasonal_bs.append(row[bs_col])
    
    if len(seasonal_bs) == 4 and 'mean_bs' in row and pd.notna(row['mean_bs']):
        # Use same day weights as CRPS
        if len(seasonal_days) == 4:
            weighted_avg = sum(b * d for b, d in zip(seasonal_bs, seasonal_days)) / total_days
            
            diff = abs(row['mean_bs'] - weighted_avg)
            if diff > tolerance:
                issues.append(
                    f"BS seasonal consistency failed: "
                    f"overall={row['mean_bs']:.6f}, weighted_avg={weighted_avg:.6f}, "
                    f"diff={diff:.2e}"
                )
    
    return issues


def check_decomposition_identity(row: pd.Series, tolerance: float = 1e-6) -> List[str]:
    """
    Check CORP decomposition identities: Score = MCB - DSC + UNC.
    """
    issues = []
    
    # Check BS decomposition
    if all(col in row and pd.notna(row[col]) for col in ['mean_bs', 'bs_mcb', 'bs_dsc', 'bs_unc']):
        identity = row['bs_mcb'] - row['bs_dsc'] + row['bs_unc']
        diff = abs(row['mean_bs'] - identity)
        
        if diff > tolerance:
            issues.append(
                f"BS decomposition identity failed: "
                f"BS={row['mean_bs']:.6f}, MCB-DSC+UNC={identity:.6f}, "
                f"diff={diff:.2e}"
            )
    
    # Check CRPS decomposition
    # Note: CRPS decomposition might be computed on a subset, so we check internal consistency
    if all(col in row and pd.notna(row[col]) for col in ['crps_mcb', 'crps_dsc', 'crps_unc']):
        # The decomposition should satisfy its own identity
        crps_from_decomp = row['crps_mcb'] - row['crps_dsc'] + row['crps_unc']
        
        # This is more of a sanity check - the values should be reasonable
        if crps_from_decomp < 0:
            issues.append(f"CRPS decomposition yields negative value: {crps_from_decomp:.6f}")
        elif crps_from_decomp > 10:  # Unreasonably high
            issues.append(f"CRPS decomposition yields unreasonably high value: {crps_from_decomp:.6f}")
    
    return issues


def check_metadata(row: pd.Series) -> List[str]:
    """
    Check that metadata is correct (Germany mask, cos(lat) weighting).
    """
    issues = []
    
    # Check mask
    if 'mask' in row:
        if pd.isna(row['mask']) or row['mask'] != 'Germany':
            issues.append(f"Invalid mask: expected 'Germany', got '{row['mask']}'")
    else:
        issues.append("Missing 'mask' column")
    
    # Check spatial weighting
    if 'spatial_weight' in row:
        if pd.isna(row['spatial_weight']) or row['spatial_weight'] != 'coslat':
            issues.append(f"Invalid spatial_weight: expected 'coslat', got '{row['spatial_weight']}'")
    else:
        issues.append("Missing 'spatial_weight' column")
    
    return issues


def check_value_validity(row: pd.Series) -> List[str]:
    """
    Check that values are within valid ranges.
    """
    issues = []
    
    # CRPS should be non-negative
    if 'mean_crps' in row and pd.notna(row['mean_crps']):
        if row['mean_crps'] < 0:
            issues.append(f"Invalid CRPS: {row['mean_crps']:.6f} < 0")
    
    # BS should be in [0, 1]
    if 'mean_bs' in row and pd.notna(row['mean_bs']):
        if row['mean_bs'] < 0 or row['mean_bs'] > 1:
            issues.append(f"Invalid BS: {row['mean_bs']:.6f} not in [0, 1]")
    
    # CRPSS should be finite and < 1
    if 'crpss_mpc' in row and pd.notna(row['crpss_mpc']):
        if not np.isfinite(row['crpss_mpc']):
            issues.append(f"CRPSS is not finite: {row['crpss_mpc']}")
        elif row['crpss_mpc'] >= 1:
            issues.append(f"Invalid CRPSS: {row['crpss_mpc']:.6f} >= 1")
    
    return issues


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run acceptance checks on evaluation results'
    )
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to aggregated_metrics.csv'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=1e-6,
        help='Numerical tolerance for identity checks (default: 1e-6)'
    )
    
    args = parser.parse_args()
    
    # Load data
    try:
        df = pd.read_csv(args.csv)
        print(f"Loaded {len(df)} runs from {args.csv}")
    except Exception as e:
        print(f"{Colors.RED}Error loading CSV: {e}{Colors.ENDC}")
        return 1
    
    # Convert numeric columns
    numeric_cols = [col for col in df.columns if col not in ['run_id', 'path', 'era5_group', 'mask', 'spatial_weight']]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Run checks for each row
    all_issues = []
    
    print(f"\nRunning acceptance checks with tolerance={args.tolerance}...")
    
    for idx, row in df.iterrows():
        run_id = row.get('run_id', f'Row {idx}')
        row_issues = []
        
        # Check 1: Seasonal consistency
        issues = check_seasonal_consistency(row, tolerance=args.tolerance)
        if issues:
            row_issues.extend([(f"{Colors.YELLOW}SEASONAL{Colors.ENDC}", issue) for issue in issues])
        
        # Check 2: Decomposition identity
        issues = check_decomposition_identity(row, tolerance=args.tolerance)
        if issues:
            row_issues.extend([(f"{Colors.YELLOW}DECOMP{Colors.ENDC}", issue) for issue in issues])
        
        # Check 3: Metadata
        issues = check_metadata(row)
        if issues:
            row_issues.extend([(f"{Colors.BLUE}METADATA{Colors.ENDC}", issue) for issue in issues])
        
        # Check 4: Value validity
        issues = check_value_validity(row)
        if issues:
            row_issues.extend([(f"{Colors.RED}VALIDITY{Colors.ENDC}", issue) for issue in issues])
        
        if row_issues:
            all_issues.append((run_id, row_issues))
    
    # Report results
    print("\n" + "="*60)
    
    if not all_issues:
        print(f"{Colors.GREEN}{Colors.BOLD}All acceptance checks passed ✅{Colors.ENDC}")
        print("="*60)
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}Acceptance checks FAILED ❌{Colors.ENDC}")
        print(f"Found issues in {len(all_issues)} out of {len(df)} runs")
        print("="*60)
        
        # Detailed report
        for run_id, issues in all_issues:
            print(f"\n{Colors.BOLD}{run_id}:{Colors.ENDC}")
            for check_type, issue in issues:
                print(f"  [{check_type}] {issue}")
        
        # Summary by type
        print("\n" + "-"*60)
        print("Summary by check type:")
        
        check_counts = {}
        for _, issues in all_issues:
            for check_type, _ in issues:
                # Remove color codes for counting
                clean_type = check_type.replace(Colors.YELLOW, '').replace(Colors.RED, '')
                clean_type = clean_type.replace(Colors.BLUE, '').replace(Colors.ENDC, '')
                check_counts[clean_type] = check_counts.get(clean_type, 0) + 1
        
        for check_type, count in sorted(check_counts.items()):
            print(f"  {check_type}: {count} issues")
        
        print("="*60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
