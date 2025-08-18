#!/usr/bin/env python
"""
Visualize the two masks used in the MSWEP UNet model to verify their correct implementation.
This script helps ensure the distinction between evaluation and loss weighting masks is clear.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.mswep_unet import MSWEPUNet

def visualize_masks():
    """Create visualizations of both masks to verify their implementation."""
    
    # Create model instance with typical parameters
    model = MSWEPUNet(
        in_channels=5,
        grid_lat=41,
        grid_lon=121,
        use_regional_focus=True,
        region_weight=1.0,
        outside_weight=0.2
    )
    
    # Extract masks
    germany_mask = model.germany_mask.numpy()
    spatial_weight_mask = model.spatial_weight_mask.numpy()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Boolean Germany Mask (for evaluation)
    im1 = ax1.imshow(germany_mask, cmap='RdBu_r', vmin=0, vmax=1)
    ax1.set_title('Germany Mask (Boolean)\nFor Evaluation Filtering Only', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitude Index')
    ax1.set_ylabel('Latitude Index')
    
    # Add grid
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Value (True=1, False=0)', rotation=270, labelpad=20)
    
    # Add text annotation
    num_true = np.sum(germany_mask)
    ax1.text(0.02, 0.98, f'Germany cells: {num_true}', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Spatial Weight Mask (for loss weighting)
    im2 = ax2.imshow(spatial_weight_mask, cmap='YlOrRd', vmin=0, vmax=1)
    ax2.set_title('Spatial Weight Mask (Float)\nFor Loss Weighting Only', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Longitude Index')
    ax2.set_ylabel('Latitude Index')
    
    # Add grid
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Weight Value', rotation=270, labelpad=20)
    
    # Count cells by weight category
    germany_cells = np.sum(np.isclose(spatial_weight_mask, 1.0))
    neighbor_cells = np.sum(np.isclose(spatial_weight_mask, 0.6, atol=0.1))
    distant_cells = np.sum(np.isclose(spatial_weight_mask, 0.2))
    
    # Add text annotation
    weight_info = f'Germany (w=1.0): {germany_cells} cells\n' + \
                  f'Neighbors (w≈0.6): {neighbor_cells} cells\n' + \
                  f'Distant (w=0.2): {distant_cells} cells'
    ax2.text(0.02, 0.98, weight_info, 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add overall title
    fig.suptitle('MSWEP UNet Masks: Clear Separation of Concerns', fontsize=16, fontweight='bold')
    
    # Add explanation text
    explanation = (
        "LEFT: Boolean mask identifying Germany region (99 cells) - used ONLY for filtering which cells are included in evaluation metrics.\n"
        "RIGHT: Float mask with gradual weights - used ONLY for weighting the loss function during training to focus on Germany while learning from neighbors."
    )
    fig.text(0.5, 0.02, explanation, ha='center', fontsize=11, style='italic', wrap=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.92)
    
    # Save figure
    output_path = 'mask_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Mask visualization saved to: {output_path}")
    
    # Also create a difference plot to highlight the neighboring region
    fig2, ax3 = plt.subplots(1, 1, figsize=(8, 6))
    
    # Create a categorical map
    category_map = np.zeros_like(spatial_weight_mask)
    category_map[np.isclose(spatial_weight_mask, 1.0)] = 2  # Germany
    category_map[np.isclose(spatial_weight_mask, 0.6, atol=0.1)] = 1  # Neighbors
    category_map[np.isclose(spatial_weight_mask, 0.2)] = 0  # Distant
    
    # Custom colormap
    from matplotlib.colors import ListedColormap
    colors = ['lightblue', 'yellow', 'darkred']
    cmap = ListedColormap(colors)
    
    im3 = ax3.imshow(category_map, cmap=cmap, vmin=0, vmax=2)
    ax3.set_title('Regional Categories for Loss Weighting', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Longitude Index')
    ax3.set_ylabel('Latitude Index')
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Create custom colorbar
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, ticks=[0, 1, 2])
    cbar3.ax.set_yticklabels(['Distant (0.2)', 'Neighbors (0.6)', 'Germany (1.0)'])
    
    plt.tight_layout()
    plt.savefig('regional_categories.png', dpi=300, bbox_inches='tight')
    print(f"Regional categories visualization saved to: regional_categories.png")
    
    plt.show()

if __name__ == "__main__":
    visualize_masks() 