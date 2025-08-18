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

# It's good practice to have these constants clearly defined or imported.
# If they are primarily used for mask creation within this MSWEPUNet,
# defining them here or passing them to __init__ is also an option.
# For now, assuming they are available as globals as per previous discussions
# (e.g., imported from mswep_evaluation.py or defined in this file if not).
# If not already imported or defined in your actual mswep_unet.py, you'll need to add:
# from models.mswep_evaluation import GERMANY_BOX_GRID_LAT_INDICES, GERMANY_BOX_GRID_LON_INDICES
# Or define them:
GERMANY_BOX_GRID_LAT_INDICES = (17, 25)  # Corresponds to ~47N to 55N
GERMANY_BOX_GRID_LON_INDICES = (75, 85)  # Corresponds to ~5E to 15E


class MSWEPUNet(nn.Module):
    """
    UNet architecture optimized for the MSWEP precipitation data.

    The model architecture follows a standard UNet with skip connections, modified for
    precipitation forecasting. Supports dynamic input channels to incorporate ERA5 
    atmospheric variables alongside precipitation data.

    Input/Output:
    - Input: N-channel tensor where:
      * Default (N=5): 3 precipitation lags (t-3, t-2, t-1) + sine/cosine of DOY
      * With ERA5: 5 base channels + 3*M ERA5 channels (M = number of ERA5 variables)
        Example: With 6 ERA5 variables (u500, u850, v500, v850, q500, q850), N=23
    - Output: Single-channel tensor for current day's precipitation (t)

    Key Features:
    1. UNet architecture with skip connections for preserving spatial information
    2. Support for regional focus with weighted loss functions (via spatial_weight_mask)
    3. Provides a boolean germany_mask for evaluation purposes
    4. Dynamic input channels for flexible integration of atmospheric predictors
    """
    def __init__(self, in_channels=5, initial_filter_size=64, kernel_size=3, do_instancenorm=True,
                 dropout=0.2, grid_lat=41, grid_lon=121, use_regional_focus=True,
                 region_weight=1.0, outside_weight=0.2, apply_transform=True, transform_probability=0.5):
        super().__init__()

        self.grid_lat = grid_lat
        self.grid_lon = grid_lon
        self.use_regional_focus = use_regional_focus
        self.region_weight = region_weight  # Weight for Germany in spatial_weight_mask
        self.outside_weight = outside_weight # Weight for distant areas in spatial_weight_mask

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

        # Center block
        self.center = nn.Sequential(
            nn.Conv2d(initial_filter_size * 8, initial_filter_size * 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(initial_filter_size * 16, initial_filter_size * 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(initial_filter_size * 16, initial_filter_size * 8, 2, stride=2, output_padding=1),
            nn.ReLU(inplace=True),
        )

        # Expanding path
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

        self.upscale1 = nn.Sequential(
            nn.ConvTranspose2d(initial_filter_size, initial_filter_size, 3, stride=(1,3), padding=(1,1)),
            nn.ReLU(inplace=True)
        )

        self.final = nn.Conv2d(initial_filter_size, 1, kernel_size=1)

        # --- Mask Initialization: Two Masks for Two Purposes ---
        # IMPORTANT: These masks serve completely different purposes and must not be confused!
        
        # 1. EVALUATION MASK: self.germany_mask (boolean)
        #    - Purpose: Filter which cells are included in metric calculations (MAE, RMSE, CRPS)
        #    - Type: Boolean tensor where True = Germany (99 cells), False = outside Germany
        #    - Usage: ONLY for filtering during metric calculation, NEVER for loss weighting
        self.germany_mask = self._create_fixed_germany_boolean_mask()

        # 2. LOSS WEIGHTING MASK: self.spatial_weight_mask (float)
        #    - Purpose: Apply different weights to loss based on proximity to Germany
        #    - Type: Float tensor with values:
        #      * region_weight (1.0) for Germany cells
        #      * ~0.6 for neighboring cells (within 3 cells of Germany)
        #      * outside_weight (0.2) for distant cells
        #    - Usage: ONLY for loss weighting during training, NEVER for metric filtering
        self.spatial_weight_mask = self._initialize_spatial_loss_weight_mask()


        # Diagnostic print to confirm the boolean Germany mask details AT CREATION
        print(f"[MSWEPUNet __init__] BOOLEAN Germany Mask (self.germany_mask) created. "
              f"Shape: {self.germany_mask.shape}, Num True values: {torch.sum(self.germany_mask).item()}")
        print(f"[MSWEPUNet __init__] Spatial LOSS WEIGHT Mask (self.spatial_weight_mask) created. "
              f"Shape: {self.spatial_weight_mask.shape}, Example weights: Germany={self.region_weight}, Outside={self.outside_weight}")

        # Add image transforms
        self.apply_transform = apply_transform
        if apply_transform:
            self.transform = transforms.RandomApply(
                torch.nn.ModuleList([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                ]), p=transform_probability
            )
        else:
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
            nn.Dropout2d(dropout, inplace=True),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, enable_concat=True, debug=False, apply_easyuq=False): # apply_easyuq not used in UNet itself
        concat_weight = 1 if enable_concat else 0

        if debug: print(f"Input shape: {x.shape}")

        x = self.transform(x)
        if debug and self.apply_transform: print(f"After transform shape: {x.shape}")

        contr_1 = self.contr_1_2(self.contr_1_1(x))
        if debug: print(f"contr_1 shape: {contr_1.shape}")
        pool = self.pool(contr_1)
        if debug: print(f"pool1 shape: {pool.shape}")

        contr_2 = self.contr_2_2(self.contr_2_1(pool))
        if debug: print(f"contr_2 shape: {contr_2.shape}")
        pool = self.pool(contr_2)
        if debug: print(f"pool2 shape: {pool.shape}")

        contr_3 = self.contr_3_2(self.contr_3_1(pool))
        if debug: print(f"contr_3 shape: {contr_3.shape}")
        pool = self.pool(contr_3)
        if debug: print(f"pool3 shape: {pool.shape}")

        contr_4 = self.contr_4_2(self.contr_4_1(pool))
        if debug: print(f"contr_4 shape: {contr_4.shape}")
        pool = self.pool(contr_4)
        if debug: print(f"pool4 shape: {pool.shape}")

        pool_4 = pool

        center = self.center(pool)
        if debug: print(f"center shape: {center.shape}")

        concat = torch.cat([center, contr_4 * concat_weight], 1)
        if debug: print(f"concat4 shape: {concat.shape}")

        expand = self.expand_4_2(self.expand_4_1(concat))
        expand = expand + F.interpolate(pool_4, size=expand.shape[2:], mode='bilinear', align_corners=False)
        if debug: print(f"expand4 shape: {expand.shape}")
        upscale = self.upscale4(expand)
        if debug: print(f"upscale4 shape: {upscale.shape}")

        concat = torch.cat([upscale, contr_3 * concat_weight], 1)
        if debug: print(f"concat3 shape: {concat.shape}")
        expand = self.expand_3_2(self.expand_3_1(concat))
        if debug: print(f"expand3 shape: {expand.shape}")
        upscale = self.upscale3(expand)
        if debug: print(f"upscale3 shape: {upscale.shape}")

        concat = torch.cat([upscale, contr_2 * concat_weight], 1)
        if debug: print(f"concat2 shape: {concat.shape}")
        expand = self.expand_2_2(self.expand_2_1(concat))
        if debug: print(f"expand2 shape: {expand.shape}")
        upscale = self.upscale2(expand)
        if debug: print(f"upscale2 shape: {upscale.shape}")

        concat = torch.cat([upscale, contr_1 * concat_weight], 1)
        if debug: print(f"concat1 shape: {concat.shape}")
        expand = self.expand_1_2(self.expand_1_1(concat))
        if debug: print(f"expand1 shape: {expand.shape}")

        upscale = self.upscale1(expand)
        if debug: print(f"final upscale shape: {upscale.shape}")

        output = self.final(upscale)
        if debug: print(f"output shape: {output.shape}")

        return output.squeeze(1)

    def _create_fixed_germany_boolean_mask(self):
        """
        Creates a boolean mask tensor of shape (grid_lat, grid_lon) where
        True indicates cells within the predefined Germany bounding box.
        This mask is fixed and used for evaluation purposes.
        """
        mask = torch.zeros(self.grid_lat, self.grid_lon, dtype=torch.bool)
        lat_min_idx, lat_max_idx = GERMANY_BOX_GRID_LAT_INDICES
        lon_min_idx, lon_max_idx = GERMANY_BOX_GRID_LON_INDICES
        mask[lat_min_idx:lat_max_idx+1, lon_min_idx:lon_max_idx+1] = True
        return mask

    def _initialize_spatial_loss_weight_mask(self):
        """
        Initialize a tensor with three discrete regions: Germany, neighboring, and distant,
        assigning different float weights to each, based on self.region_weight and self.outside_weight.
        This mask is intended for use in spatially weighted loss functions during training.

        Returns:
            torch.Tensor: A mask tensor of shape (grid_lat, grid_lon) with region-based float weights.
        """
        # Create a base mask with lower weights for distant regions
        weight_mask = torch.ones(self.grid_lat, self.grid_lon, dtype=torch.float32) * self.outside_weight

        # Define Germany's grid boundaries using the global/imported constants
        lat_min_idx = GERMANY_BOX_GRID_LAT_INDICES[0]
        lat_max_idx = GERMANY_BOX_GRID_LAT_INDICES[1]
        lon_min_idx = GERMANY_BOX_GRID_LON_INDICES[0]
        lon_max_idx = GERMANY_BOX_GRID_LON_INDICES[1]

        # Calculate intermediate weight for neighboring regions
        # This can be a fixed intermediate value or derived.
        # For simplicity, let's make it halfway if distinct, or same as Germany if outside_weight is close to region_weight.
        if abs(self.region_weight - self.outside_weight) < 1e-5: # if region and outside weights are the same
            neighbor_weight = self.region_weight
        else:
            neighbor_weight = (self.region_weight + self.outside_weight) / 2.0 # A simple intermediate


        # Set weights for each region
        for i in range(self.grid_lat):
            for j in range(self.grid_lon):
                is_in_germany = (lat_min_idx <= i <= lat_max_idx) and \
                                (lon_min_idx <= j <= lon_max_idx)

                if is_in_germany:
                    weight_mask[i, j] = self.region_weight
                elif self.use_regional_focus: # Only define neighbors if regional focus is on
                    # Calculate Chebyshev distance (max of lat/lon distances from the box boundary)
                    dist_lat = 0
                    if i < lat_min_idx: dist_lat = lat_min_idx - i
                    elif i > lat_max_idx: dist_lat = i - lat_max_idx

                    dist_lon = 0
                    if j < lon_min_idx: dist_lon = lon_min_idx - j
                    elif j > lon_max_idx: dist_lon = j - lon_max_idx

                    chebyshev_dist = max(dist_lat, dist_lon)

                    if chebyshev_dist <= 3:  # Within 3 cells (inclusive of the 3rd cell away)
                        weight_mask[i, j] = neighbor_weight
                    # Else, it keeps self.outside_weight from initialization

        return weight_mask

    def _get_boolean_region_masks(self, spatial_loss_weight_mask, device):
        """
        DEPRECATED/REVISED: This function is no longer the primary source for self.germany_mask.
        It can be used to derive other boolean masks (neighbor, distant) from the
        spatial_loss_weight_mask if needed, but self.germany_mask is now fixed.

        Args:
            spatial_loss_weight_mask (torch.Tensor): The float weight mask from _initialize_spatial_loss_weight_mask.
            device (torch.device): Device to place masks on.

        Returns:
            tuple: (boolean_germany_mask_derived, boolean_neighbor_mask, boolean_distant_mask)
                   where boolean_germany_mask_derived is based on region_weight in spatial_loss_weight_mask.
        """
        epsilon = 1e-5

        # This derives a Germany mask based on the actual float values in spatial_loss_weight_mask
        # It might differ from self.germany_mask if weights are set unexpectedly.
        boolean_germany_mask_derived = torch.isclose(
            spatial_loss_weight_mask,
            torch.tensor(self.region_weight, dtype=spatial_loss_weight_mask.dtype, device=spatial_loss_weight_mask.device),
            atol=epsilon
        ).to(device)
        
        calculated_neighbor_weight = (self.region_weight + self.outside_weight) / 2.0
        if abs(self.region_weight - self.outside_weight) < 1e-5:
            calculated_neighbor_weight = self.region_weight

        boolean_neighbor_mask = torch.isclose(
            spatial_loss_weight_mask,
            torch.tensor(calculated_neighbor_weight, dtype=spatial_loss_weight_mask.dtype, device=spatial_loss_weight_mask.device),
            atol=epsilon
        ).to(device)

        boolean_distant_mask = torch.isclose(
            spatial_loss_weight_mask,
            torch.tensor(self.outside_weight, dtype=spatial_loss_weight_mask.dtype, device=spatial_loss_weight_mask.device),
            atol=epsilon
        ).to(device)

        # Ensure masks are mutually exclusive if derived from distinct weights
        boolean_neighbor_mask = boolean_neighbor_mask & (~boolean_germany_mask_derived)
        boolean_distant_mask = boolean_distant_mask & (~boolean_germany_mask_derived) & (~boolean_neighbor_mask)
        
        # This derived mask should ideally match self.germany_mask if region_weight is unique
        # and correctly identifies the Germany region in spatial_loss_weight_mask.
        # For safety, self.germany_mask is now set independently.
        print(f"    [_get_boolean_region_masks] Derived Boolean Germany count from spatial_loss_weights: {torch.sum(boolean_germany_mask_derived).item()}")

        return boolean_germany_mask_derived, boolean_neighbor_mask, boolean_distant_mask


if __name__ == "__main__":
    # Test the model with different input channel configurations
    print("="*60)
    print("Testing MSWEPUNet with various input configurations")
    print("="*60)
    
    # Test 1: Default configuration (5 channels - no ERA5)
    print("\n1. DEFAULT CONFIGURATION (No ERA5)")
    model_default = MSWEPUNet(in_channels=5, grid_lat=41, grid_lon=121,
                             use_regional_focus=True, region_weight=1.0, outside_weight=0.2)
    print(f"  Input channels: 5 (3 precip lags + 2 seasonality)")
    print(f"  Germany mask sum: {torch.sum(model_default.germany_mask).item()}")
    print(f"  Spatial weight in Germany (17,75): {model_default.spatial_weight_mask[17, 75].item()}")
    print(f"  Spatial weight near Germany (16,75): {model_default.spatial_weight_mask[16, 75].item()}")
    print(f"  Spatial weight far from Germany (0,0): {model_default.spatial_weight_mask[0, 0].item()}")
    
    x_default = torch.randn(1, 5, 41, 121)
    output_default = model_default(x_default)
    print(f"  Input shape: {x_default.shape} -> Output shape: {output_default.shape}")
    
    # Test 2: With ERA5 configuration (23 channels)
    print("\n2. WITH ERA5 CONFIGURATION")
    # 5 base channels + 3 lags × 6 ERA5 variables (u500, u850, v500, v850, q500, q850) = 23 channels
    model_era5 = MSWEPUNet(in_channels=23, grid_lat=41, grid_lon=121,
                          use_regional_focus=True, region_weight=1.0, outside_weight=0.2)
    print(f"  Input channels: 23 (5 base + 18 ERA5)")
    print(f"  ERA5 variables: 6 (u500, u850, v500, v850, q500, q850)")
    print(f"  Germany mask sum: {torch.sum(model_era5.germany_mask).item()}")
    
    x_era5 = torch.randn(1, 23, 41, 121)
    output_era5 = model_era5(x_era5)
    print(f"  Input shape: {x_era5.shape} -> Output shape: {output_era5.shape}")
    
    # Test 3: Custom ERA5 configuration (e.g., 3 variables at 4 pressure levels)
    print("\n3. CUSTOM ERA5 CONFIGURATION")
    # 5 base + 3 lags × (3 vars × 4 levels) = 5 + 36 = 41 channels
    n_custom_channels = 5 + 3 * (3 * 4)  # 41 channels
    model_custom = MSWEPUNet(in_channels=n_custom_channels, grid_lat=41, grid_lon=121,
                            use_regional_focus=True, region_weight=1.0, outside_weight=1.0)
    print(f"  Input channels: {n_custom_channels} (5 base + 36 ERA5)")
    print(f"  ERA5 configuration: 3 variables × 4 pressure levels × 3 lags")
    print(f"  Germany mask sum: {torch.sum(model_custom.germany_mask).item()}")
    
    x_custom = torch.randn(1, n_custom_channels, 41, 121)
    output_custom = model_custom(x_custom)
    print(f"  Input shape: {x_custom.shape} -> Output shape: {output_custom.shape}")
    
    print("\n" + "="*60)
    print("All tests passed! Model handles dynamic input channels correctly.")