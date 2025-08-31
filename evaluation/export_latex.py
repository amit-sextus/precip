#!/usr/bin/env python3
"""
Export evaluation results to LaTeX tables with booktabs formatting.

Usage:
    python -m evaluation.export_latex --csv evaluation/aggregated_metrics.csv --out evaluation/tables/results.tex
    python -m evaluation.export_latex --csv evaluation/aggregated_metrics.csv --out evaluation/tables/regional.tex --select 'era5_group=="none"'
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np


def format_number(val, decimals=3):
    """Format a number to specified decimals, handling NaN."""
    if pd.isna(val):
        return '---'
    try:
        return f"{float(val):.{decimals}f}"
    except:
        return str(val)


def create_seasonal_summary_table(df: pd.DataFrame, caption: str = None) -> str:
    """
    Create Table 1: Per-run seasonal summary.
    
    Columns: Run, Outside w, ERA5 group, Overall CRPS, CRPSS vs MPC, 
             and seasonal CRPS (DJF/MAM/JJA/SON).
    """
    lines = []
    
    # LaTeX preamble
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\small")
    
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    else:
        lines.append("\\caption{Per-run seasonal CRPS summary}")
    
    # Begin tabular
    lines.append("\\begin{tabular}{llccccccc}")
    lines.append("\\toprule")
    
    # Header
    lines.append("Run ID & Outside $w$ & ERA5 & Overall & CRPSS & \\multicolumn{4}{c}{Seasonal CRPS} \\\\")
    lines.append("& & Group & CRPS & vs MPC & DJF & MAM & JJA & SON \\\\")
    lines.append("\\midrule")
    
    # Sort by mean CRPS for better presentation
    df_sorted = df.sort_values('mean_crps')
    
    # Data rows
    for idx, row in df_sorted.iterrows():
        # Shorten run_id if too long
        run_id = row['run_id']
        if len(run_id) > 20:
            run_id = run_id[:17] + '...'
        
        # Format values
        outside_w = format_number(row.get('outside_weight', np.nan), 1)
        era5_group = row.get('era5_group', '---')
        overall_crps = format_number(row.get('mean_crps', np.nan))
        crpss = format_number(row.get('crpss_mpc', np.nan))
        
        # Seasonal values
        seasonal_vals = []
        for season in ['djf', 'mam', 'jja', 'son']:
            col = f'crps_{season}'
            val = format_number(row.get(col, np.nan))
            seasonal_vals.append(val)
        
        # Build row
        row_str = f"{run_id} & {outside_w} & {era5_group} & {overall_crps} & {crpss}"
        for val in seasonal_vals:
            row_str += f" & {val}"
        row_str += " \\\\"
        
        lines.append(row_str)
    
    # Footer
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    
    # Footnote
    lines.append("\\vspace{0.5em}")
    lines.append("\\footnotesize")
    lines.append("\\textit{Note}: All values computed over Germany mask with cos(lat) area-weighting. ")
    lines.append("PoP threshold: 0.2 mm.")
    
    lines.append("\\end{table}")
    
    return '\n'.join(lines)


def create_decomposition_table(df: pd.DataFrame, run_id: str = None) -> str:
    """
    Create Table 2: CORP decomposition for a chosen run.
    
    Shows: Score, MCB, DSC, UNC, Identity check for BS and CRPS.
    """
    lines = []
    
    # Select run
    if run_id:
        run_data = df[df['run_id'] == run_id]
        if run_data.empty:
            return f"% Run '{run_id}' not found in data"
        row = run_data.iloc[0]
    else:
        # Use best run (lowest CRPS)
        best_idx = df['mean_crps'].idxmin()
        row = df.loc[best_idx]
        run_id = row['run_id']
    
    # LaTeX preamble
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(f"\\caption{{CORP decomposition for {run_id}}}")
    
    # Begin tabular
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    
    # Header
    lines.append("Score & Value & MCB & DSC & UNC & Identity Check \\\\")
    lines.append("& & (Miscalib.) & (Discrim.) & (Uncert.) & (MCB$-$DSC$+$UNC$-$Score) \\\\")
    lines.append("\\midrule")
    
    # Brier Score row
    bs = row.get('mean_bs', np.nan)
    bs_mcb = row.get('bs_mcb', np.nan)
    bs_dsc = row.get('bs_dsc', np.nan)
    bs_unc = row.get('bs_unc', np.nan)
    
    if all(pd.notna([bs, bs_mcb, bs_dsc, bs_unc])):
        identity = bs_mcb - bs_dsc + bs_unc - bs
        bs_row = f"Brier Score (0.2 mm) & {format_number(bs)} & {format_number(bs_mcb)} & "
        bs_row += f"{format_number(bs_dsc)} & {format_number(bs_unc)} & "
        bs_row += f"{format_number(identity, 6)} \\\\"
        lines.append(bs_row)
    else:
        lines.append("Brier Score (0.2 mm) & --- & --- & --- & --- & --- \\\\")
    
    # CRPS row
    crps = row.get('mean_crps', np.nan)
    crps_mcb = row.get('crps_mcb', np.nan)
    crps_dsc = row.get('crps_dsc', np.nan)
    crps_unc = row.get('crps_unc', np.nan)
    
    # Note: CRPS from decomposition might be on a subset
    if all(pd.notna([crps_mcb, crps_dsc, crps_unc])):
        # Compute CRPS from decomposition
        crps_decomp = crps_mcb - crps_dsc + crps_unc
        identity = crps_mcb - crps_dsc + crps_unc - crps_decomp
        crps_row = f"CRPS$^*$ & {format_number(crps_decomp)} & {format_number(crps_mcb)} & "
        crps_row += f"{format_number(crps_dsc)} & {format_number(crps_unc)} & "
        crps_row += f"{format_number(identity, 6)} \\\\"
        lines.append(crps_row)
    else:
        lines.append("CRPS & --- & --- & --- & --- & --- \\\\")
    
    # Footer
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    
    # Footnotes
    lines.append("\\vspace{0.5em}")
    lines.append("\\footnotesize")
    lines.append("\\textit{Note}: Identity check should be $\\approx 0$ (tolerance: 1e-6). ")
    if all(pd.notna([crps_mcb, crps_dsc, crps_unc])):
        lines.append("$^*$CRPS decomposition computed on diagnostic subset.")
    
    lines.append("\\end{table}")
    
    return '\n'.join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Export evaluation results to LaTeX tables'
    )
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to aggregated_metrics.csv'
    )
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='Output LaTeX file path'
    )
    parser.add_argument(
        '--select',
        type=str,
        default=None,
        help='Optional pandas query to filter rows (e.g., \'era5_group=="none"\')'
    )
    parser.add_argument(
        '--table-type',
        type=str,
        choices=['seasonal', 'decomposition', 'both'],
        default='seasonal',
        help='Type of table to generate'
    )
    parser.add_argument(
        '--run-id',
        type=str,
        default=None,
        help='Specific run ID for decomposition table'
    )
    parser.add_argument(
        '--caption',
        type=str,
        default=None,
        help='Custom table caption'
    )
    
    args = parser.parse_args()
    
    # Load data
    try:
        df = pd.read_csv(args.csv)
        print(f"Loaded {len(df)} runs from {args.csv}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return 1
    
    # Apply selection filter if provided
    if args.select:
        try:
            df_filtered = df.query(args.select)
            print(f"Filtered to {len(df_filtered)} runs using: {args.select}")
            df = df_filtered
        except Exception as e:
            print(f"Error applying filter: {e}")
            return 1
    
    if df.empty:
        print("No data to export after filtering")
        return 1
    
    # Convert numeric columns
    numeric_cols = ['outside_weight', 'mean_crps', 'mean_bs', 'crpss_mpc',
                   'crps_djf', 'crps_mam', 'crps_jja', 'crps_son',
                   'bs_djf', 'bs_mam', 'bs_jja', 'bs_son',
                   'crpss_djf', 'crpss_mam', 'crpss_jja', 'crpss_son',
                   'bs_mcb', 'bs_dsc', 'bs_unc',
                   'crps_mcb', 'crps_dsc', 'crps_unc']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Generate table(s)
    tables = []
    
    if args.table_type in ['seasonal', 'both']:
        table = create_seasonal_summary_table(df, caption=args.caption)
        tables.append(table)
    
    if args.table_type in ['decomposition', 'both']:
        table = create_decomposition_table(df, run_id=args.run_id)
        tables.append(table)
    
    # Write output
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    with open(args.out, 'w') as f:
        # LaTeX document preamble (commented out - assume included in main document)
        f.write("% Generated by evaluation.export_latex\n")
        f.write("% Include in your LaTeX document with \\input{" + args.out + "}\n")
        f.write("% Requires: \\usepackage{booktabs}\n\n")
        
        # Write tables
        f.write('\n\n'.join(tables))
    
    print(f"Wrote LaTeX table(s) to: {args.out}")
    
    # Print example usage for main document
    print("\nTo use in your LaTeX document:")
    print("\\usepackage{booktabs}")
    print(f"\\input{{{args.out}}}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
