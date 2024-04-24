from __future__ import annotations

import torch
from torch.nn.functional import interpolate

from .exposed_dynamicrafter import Image2VideoExposed


class ExposedDynamiCrafter(torch.nn.Module):
    def __init__(
        self,
        time_step=12,
        resolution = "320_512",
        output="dense",
        layer=1,
        return_multilayer=False,
        batch_size = 8, 
        is_single_image= True
    ):
        super().__init__()
        assert output in ["gap", "dense"], "Only supports gap or dense output"

        print("is_single_image", is_single_image)   

        self.output = output
        self.time_step = time_step
        model_id = 'dynamicrafter_'+resolution.split('_')[1]+'_interp_v1'
        self.checkpoint_name = model_id + f"_noise-{time_step}"
        self.patch_size = 16
        self.batch_size = batch_size
        self.feature_extractor = Image2VideoExposed(resolution=resolution, is_single_image=is_single_image)
        # self.up_ft_index = [0, 2, 4, 6, 8, 10, 11]
        self.up_ft_index = [0, 4, 8, 11]
        # assert layer in [-1, 0, 2, 4, 6, 8, 10, 11]

        # feat_dims = [1280, 1280, 1280, 640, 640, 320, 320]
        feat_dims = [1280, 1280, 640, 320]
        # multilayers = [0, 2, 4, 6, 8, 10, 11]
        multilayers = [0, 4, 8, 11]

        if return_multilayer:
            self.feat_dim = feat_dims
            self.multilayers = multilayers
        else:
            layer = multilayers[-1] if layer == -1 else layer
            self.feat_dim = feat_dims[layer]
            self.multilayers = [multilayers[layer]]

        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)

    def forward(self, images, categories=None, prompts=None):
        spatial = []
        batch_size = self.batch_size

        # handle prompts
        assert categories is None or prompts is None, "Cannot be both"
        if categories:
            prompts = [f"a photo of a {_c}" for _c in categories]
        elif prompts is None:
            prompts = ["" for _ in range(batch_size)]

        assert len(prompts) == batch_size
        # print("images shape", images.shape)        
        spatial = self.feature_extractor.get_features(
            images, prompt=prompts, exposed_timestep_index=self.time_step-1, batch_size = self.batch_size 
        )
        h, w = images.shape[2] // self.patch_size, images.shape[3] // self.patch_size
        spatial = [spatial[i] for i in self.multilayers]

        assert self.output in ["gap", "dense"]
        if self.output == "gap":
            spatial = [x.mean(dim=(2, 3)) for x in spatial]
        elif self.output == "dense":
            spatial = [interpolate(x.contiguous(), (h, w)) for x in spatial]
        return spatial[0] if len(spatial) == 1 else spatial
