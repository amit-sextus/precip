"""
UNet model for MSWEP precipitation forecasting.

This module implements a UNet architecture optimized for precipitation forecasting
using MSWEP data, with appropriate handling for the specific grid sizes and
precipitation characteristics.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms






class MSWEPUNet(nn.Module):
    """
    UNet architecture optimized for the MSWEP precipitation data.
    
    The model architecture follows a standard UNet with skip connections, modified for
    precipitation forecasting.
    
    Input/Output:
    - Input: 5-channel tensor representing precipitation at days t-3, t-2, t-1, and sine/cosine of DOY
    - Output: Single-channel tensor for current day's precipitation (t)
    
    Key Features:
    1. UNet architecture with skip connections for preserving spatial information
    2. Support for regional focus with weighted loss functions (via germany_mask)
    """
    def __init__(self, in_channels=5, initial_filter_size=64, kernel_size=3, do_instancenorm=True, 
                 dropout=0.2, grid_lat=41, grid_lon=121, use_regional_focus=True, 
                 region_weight=1.0, outside_weight=0.2, apply_transform=True, transform_probability=0.5):
        super().__init__()

        # Contracting path
        self.contr_1_1 = self.contract(in_channels, initial_filter_size, kernel_size, stride=(1,3),
                                       instancenorm=do_instancenorm, dropout=dropout)
        self.contr_1_2 = self.contract(initial_filter_size, initial_filter_size, kernel_size, 
                                       instancenorm=do_instancenorm, dropout=dropout)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.contr_2_1 = self.contract(initial_filter_size, initial_filter_size * 2, kernel_size,
                                       instancenorm=do_instancenorm, dropout=dropout)
        self.contr_2_2 = self.contract(initial_filter_size * 2, initial_filter_size * 2, kernel_size,
                                       instancenorm=do_instancenorm, dropout=dropout)

        self.contr_3_1 = self.contract(initial_filter_size * 2, initial_filter_size * 4, kernel_size,
                                       instancenorm=do_instancenorm, dropout=dropout)
        self.contr_3_2 = self.contract(initial_filter_size * 4, initial_filter_size * 4, kernel_size,
                                       instancenorm=do_instancenorm, dropout=dropout)

        self.contr_4_1 = self.contract(initial_filter_size * 4, initial_filter_size * 8, kernel_size,
                                       instancenorm=do_instancenorm, dropout=dropout)
        self.contr_4_2 = self.contract(initial_filter_size * 8, initial_filter_size * 8, kernel_size,
                                       instancenorm=do_instancenorm, dropout=dropout)

        # Center block - fix output size to match contr_4 (5x5)
        # Using output_padding=1 to ensure dimensions align properly
        self.center = nn.Sequential(
            nn.Conv2d(initial_filter_size * 8, initial_filter_size * 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(initial_filter_size * 16, initial_filter_size * 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(initial_filter_size * 16, initial_filter_size * 8, 2, stride=2, output_padding=1),
            nn.ReLU(inplace=True),
        )

        # Expanding path - adjusted for proper channel counts after concatenation
        self.expand_4_1 = self.expand(initial_filter_size * 16, initial_filter_size * 8, dropout=dropout)
        self.expand_4_2 = self.expand(initial_filter_size * 8, initial_filter_size * 8, dropout=dropout)
        self.upscale4 = nn.ConvTranspose2d(initial_filter_size * 8, initial_filter_size * 4, 2, stride=2)

        self.expand_3_1 = self.expand(initial_filter_size * 8, initial_filter_size * 4, dropout=dropout)
        self.expand_3_2 = self.expand(initial_filter_size * 4, initial_filter_size * 4, dropout=dropout)
        self.upscale3 = nn.ConvTranspose2d(initial_filter_size * 4, initial_filter_size * 2, 2, stride=2)

        self.expand_2_1 = self.expand(initial_filter_size * 4, initial_filter_size * 2, dropout=dropout)
        self.expand_2_2 = self.expand(initial_filter_size * 2, initial_filter_size * 2, dropout=dropout)
        self.upscale2 = nn.ConvTranspose2d(initial_filter_size * 2, initial_filter_size, 2, stride=2, output_padding=(1,1))

        self.expand_1_1 = self.expand(initial_filter_size * 2, initial_filter_size, dropout=dropout)
        self.expand_1_2 = self.expand(initial_filter_size, initial_filter_size, dropout=dropout)
        
        # Asymmetric stride for the final upsampling remains the same
        self.upscale1 = nn.Sequential(
            nn.ConvTranspose2d(initial_filter_size, initial_filter_size, 3, stride=(1,3), padding=(1,1)),
            nn.ReLU(inplace=True)
        )
        
        # Output layer - single channel for precipitation forecast
        self.final = nn.Conv2d(initial_filter_size, 1, kernel_size=1)

        # Initialize Germany mask for focused loss calculation
        self.grid_lat = grid_lat
        self.grid_lon = grid_lon
        self.use_regional_focus = use_regional_focus  # Flag to enable/disable regional focus
        self.region_weight = region_weight  # Weight for target region (Germany)
        self.outside_weight = outside_weight  # Weight for areas outside target region
        self.germany_mask = self._initialize_germany_mask()
        
        # Add image transforms 
        self.apply_transform = apply_transform
        if apply_transform:
            self.transform = transforms.RandomApply(
                torch.nn.ModuleList([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                ]), p=transform_probability
            )
        else:
            # Create identity transform when transform is disabled
            self.transform = nn.Identity()
        
        
     
    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3, stride=1, instancenorm=True, dropout=0.0):
        if instancenorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, stride=stride),
                nn.Dropout2d(dropout, inplace=True),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(inplace=True)
            )
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, stride=stride),
                nn.LeakyReLU(inplace=True)
            )
        return layer

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=3, dropout=0.0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
            # Replace standard Dropout2d with spatial dropout that preserves meteorological patterns
            nn.Dropout2d(dropout, inplace=True),  # Drops entire feature maps, better for spatial data
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, enable_concat=True, debug=False, apply_easyuq=False):
        concat_weight = 1 if enable_concat else 0

        if debug:
            print(f"Input shape: {x.shape}")
        
        # Apply transform (Gaussian blur or Identity)
        x = self.transform(x)
        if debug and self.apply_transform:
            print(f"After transform shape: {x.shape}")
        
        contr_1 = self.contr_1_2(self.contr_1_1(x))
        if debug:
            print(f"contr_1 shape: {contr_1.shape}")
        
        pool = self.pool(contr_1)
        if debug:
            print(f"pool1 shape: {pool.shape}")

        contr_2 = self.contr_2_2(self.contr_2_1(pool))
        if debug:
            print(f"contr_2 shape: {contr_2.shape}")
        
        pool = self.pool(contr_2)
        if debug:
            print(f"pool2 shape: {pool.shape}")

        contr_3 = self.contr_3_2(self.contr_3_1(pool))
        if debug:
            print(f"contr_3 shape: {contr_3.shape}")
        
        pool = self.pool(contr_3)
        if debug:
            print(f"pool3 shape: {pool.shape}")
        
        contr_4 = self.contr_4_2(self.contr_4_1(pool))
        if debug:
            print(f"contr_4 shape: {contr_4.shape}")
        
        pool = self.pool(contr_4)
        if debug:
            print(f"pool4 shape: {pool.shape}")
        
        # Store the pool output for residual connection
        pool_4 = pool  # After 4th pooling
        
        center = self.center(pool)
        if debug:
            print(f"center shape: {center.shape}")
            print(f"contr_4 shape for skip: {contr_4.shape}")
            if center.shape[2:] != contr_4.shape[2:]:
                print(f"WARNING: Dimension mismatch at level 4 - center: {center.shape[2:]}, contr_4: {contr_4.shape[2:]}")
        
        # Concatenate along channel dimension
        concat = torch.cat([center, contr_4 * concat_weight], 1)
        if debug:
            print(f"concat4 shape: {concat.shape}")
        
        # Level 4 upscaling
        expand = self.expand_4_2(self.expand_4_1(concat))
        expand = expand + F.interpolate(pool_4, size=expand.shape[2:], mode='bilinear', align_corners=False)
        if debug:
            print(f"expand4 shape: {expand.shape}")
        
        upscale = self.upscale4(expand)
        if debug:
            print(f"upscale4 shape: {upscale.shape}")
            print(f"contr_3 shape for skip: {contr_3.shape}")
            if upscale.shape[2:] != contr_3.shape[2:]:
                print(f"WARNING: Dimension mismatch at level 3 - upscale4: {upscale.shape[2:]}, contr_3: {contr_3.shape[2:]}")
        
        # Level 3 upscaling
        concat = torch.cat([upscale, contr_3 * concat_weight], 1)
        if debug:
            print(f"concat3 shape: {concat.shape}")
        
        expand = self.expand_3_2(self.expand_3_1(concat))
        if debug:
            print(f"expand3 shape: {expand.shape}")
        
        upscale = self.upscale3(expand)
        if debug:
            print(f"upscale3 shape: {upscale.shape}")
            print(f"contr_2 shape for skip: {contr_2.shape}")
            if upscale.shape[2:] != contr_2.shape[2:]:
                print(f"WARNING: Dimension mismatch at level 2 - upscale3: {upscale.shape[2:]}, contr_2: {contr_2.shape[2:]}")
        
        # Level 2 upscaling
        concat = torch.cat([upscale, contr_2 * concat_weight], 1)
        if debug:
            print(f"concat2 shape: {concat.shape}")
        
        expand = self.expand_2_2(self.expand_2_1(concat))
        if debug:
            print(f"expand2 shape: {expand.shape}")
        
        upscale = self.upscale2(expand)
        if debug:
            print(f"upscale2 shape: {upscale.shape}")
            print(f"contr_1 shape for skip: {contr_1.shape}")
            if upscale.shape[2:] != contr_1.shape[2:]:
                print(f"WARNING: Dimension mismatch at level 1 - upscale2: {upscale.shape[2:]}, contr_1: {contr_1.shape[2:]}")
        
        # Level 1 upscaling
        concat = torch.cat([upscale, contr_1 * concat_weight], 1)
        if debug:
            print(f"concat1 shape: {concat.shape}")
        
        expand = self.expand_1_2(self.expand_1_1(concat))
        if debug:
            print(f"expand1 shape: {expand.shape}")
        
        # Final upscale
        upscale = self.upscale1(expand)
        if debug:
            print(f"final upscale shape: {upscale.shape}")
        
        # Final 1x1 convolution
        output = self.final(upscale)
        if debug:
            print(f"output shape: {output.shape}")
        
        
        return output.squeeze(1)

   

    def _initialize_germany_mask(self):
        """
        Initialize a mask tensor with three discrete regions: Germany, neighboring, and distant.
        
        Creates a spatial weight mask over the domain with:
        - Full weight (region_weight, default 1.0) for Germany
        - Intermediate weight for neighboring regions (within 3 cells of Germany)
        - Lower weight (outside_weight, configurable from command line) for distant regions
        
        This approach is scientifically sound for atmospheric processes as it:
        1. Maintains physical consistency for meteorological systems crossing boundaries
        2. Properly weights forecasts across the entire domain
        3. Follows established practices in regional numerical weather prediction
        
        Returns:
            torch.Tensor: A mask tensor of shape (41, 121) with region-based weights
        """
        # Create a base mask with lower weights for distant regions
        mask = torch.ones(self.grid_lat, self.grid_lon) * self.outside_weight
        
        # Define Germany's grid boundaries - hard-coded for scientific consistency
        # Based on approximate mapping of Germany at 47°-55°N and 5°-15°E in the grid
        lat_min, lat_max = 17, 25  # Germany latitude range
        lon_min, lon_max = 75, 85  # Germany longitude range
        
        # Calculate intermediate weight for neighboring regions
        # This creates a smooth transition between Germany and distant regions
        neighbor_weight = self.region_weight * 0.7 + self.outside_weight * 0.3
        
        # Set weights for each region
        for i in range(self.grid_lat):
            for j in range(self.grid_lon):
                # Calculate distance from Germany in grid cells
                dist_lat = max(0, lat_min - i, i - lat_max)
                dist_lon = max(0, lon_min - j, j - lon_max)
                dist = (dist_lat**2 + dist_lon**2)**0.5
                
                # Apply discrete region-based weights
                if dist == 0:
                    # Inside Germany - always use region_weight (1.0)
                    mask[i, j] = self.region_weight
                elif dist <= 3:
                    # Neighboring regions (within 3 cells)
                    mask[i, j] = neighbor_weight
        
        return mask
    
    def get_region_masks(self, weight_mask, device):
        """
        Create region masks based on dynamic weight thresholds.
        
        Args:
            weight_mask: The weight mask tensor
            device: Device to place masks on
            
        Returns:
            Three boolean masks for Germany, neighboring, and distant regions
        """
        # Calculate thresholds based on actual weights
        germany_threshold = self.region_weight * 0.9  # 90% of region_weight
        neighbor_threshold = self.outside_weight * 1.5  # 150% of outside_weight
        
        # Create masks with dynamic thresholds
        germany_mask = (weight_mask > germany_threshold).to(device)
        neighbor_mask = ((weight_mask <= germany_threshold) & (weight_mask >= neighbor_threshold)).to(device)
        distant_mask = (weight_mask < neighbor_threshold).to(device)
        
        return germany_mask, neighbor_mask, distant_mask
    
   

if __name__ == "__main__":
    # Test the model
    model = MSWEPUNet()
    x = torch.randn(1, 5, 41, 121)
    output = model(x)
    print(output.shape)