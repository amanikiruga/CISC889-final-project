from __future__ import annotations

import torch
from torch.nn.functional import interpolate

from .dynamicrafter_hyperfeatures_models import DiffusionHyperfeatures, AggregationNetwork


class DynamiCrafterHyperfeatures(torch.nn.Module):
    def __init__(
        self,
        resolution = "320_512",
        output="dense",
        batch_size = 8, 
        is_single_image= True
    ):
        super().__init__()
        assert output in ["gap", "dense"], "Only supports gap or dense output"

        print("is_single_image", is_single_image)   

        self.output = output
        model_id = 'dynamicrafter_'+resolution.split('_')[1]+'_interp_v1'
        
        self.patch_size = 16
        self.batch_size = batch_size
        self.up_ft_index = [0, 1, 2 , 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.multilayers = [0, 1, 2 , 3, 4, 5, 6, 7, 8, 9, 10, 11]

        # self.timesteps = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
        self.timesteps = [5, 10, 15, 20, 25, 30, 35, 40, 45]
        # self.timesteps = [5, 15, 25, 30, 35, 45]
        self.feat_dim = 384 + 3 # 3 is in order to concatenate the input images
        feat_dims_pre_agg = [1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 640, 320, 320, 320]

        # define layer name (for logging)
        self.feature_extractor = DiffusionHyperfeatures(timesteps= self.timesteps, feat_dims= feat_dims_pre_agg, resolution=resolution, is_single_image=is_single_image)
        self.aggregation_module = AggregationNetwork(feature_dims = feat_dims_pre_agg, device="cuda", save_timestep=self.timesteps)
        self.layer = "all-layers"#"-".join(str(_x) for _x in self.multilayers)
        self.checkpoint_name = model_id# + f"_noise-{'-'.join([str(_x) for _x in self.timesteps])}"

    def forward(self, images, categories=None, prompts=None):
        batch_size = self.batch_size

        # handle prompts
        assert categories is None or prompts is None, "Cannot be both"
        if categories:
            prompts = [f"a photo of a {_c}" for _c in categories]
        elif prompts is None:
            prompts = ["" for _ in range(batch_size)]

        assert len(prompts) == batch_size
        print("images shape", images.shape)        
        # h, w = images.shape[2] // self.patch_size, images.shape[3] // self.patch_size
        h, w = images.shape[2] , images.shape[3] 

        with torch.no_grad(): 
            _,feats = self.feature_extractor.get_hyperfeatures(
                images, prompt=prompts, output_size=(40, 64), verbose = False
            )
        spatial = self.aggregation_module(feats) # b, self.feat_dims, h, w
        spatial = interpolate(spatial, size=(h, w), mode="bilinear", align_corners=False)

        # concatenate the input images
        spatial = torch.cat([images, spatial], dim=1)
        return spatial