from __future__ import annotations

import torch
from torch.nn.functional import interpolate
import sys 
sys.path.append('/work/ececis_research/peace/dino-diffusion/diffusion_hyperfeatures')
from omegaconf import OmegaConf
from archs.diffusion_extractor import DiffusionExtractor
from archs.aggregation_network import AggregationNetwork
from archs.stable_diffusion.resnet import collect_dims

def load_models(config_path, device="cuda", batch_size = 1, pretrained=False):
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    config["batch_size"] = batch_size
    if pretrained: 
        weights = torch.load(config["weights_path"], map_location="cpu")
        config.update(weights["config"])
    if config.get("flip_timesteps", False):
        config["save_timestep"] = config["save_timestep"][::-1]

    # dims is the channel dim for each layer (12 dims for Layers 1-12)
    # idxs is the (block, sub-block) index for each layer (12 idxs for Layers 1-12)
    diffusion_extractor = DiffusionExtractor(config, device)
    dims = collect_dims(diffusion_extractor.unet, idxs=diffusion_extractor.idxs)
    aggregation_network = AggregationNetwork(
        projection_dim=config["projection_dim"],
        feature_dims=dims,
        device=device,
        save_timestep=config["save_timestep"],
        num_timesteps=config["num_timesteps"],
    )
    if pretrained: 
        aggregation_network.load_state_dict(weights["aggregation_network"])
    return config, diffusion_extractor, aggregation_network


class StableDiffusionHyperfeatures(torch.nn.Module):
    def __init__(
        self,
        resolution = "320_512",
        output="dense",
        batch_size = 8, 
        config_path = "/work/ececis_research/peace/dino-diffusion/diffusion_hyperfeatures/configs/train.yaml",
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

        self.timesteps = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
        # self.timesteps = [5, 15, 25, 30, 35, 45]
        self.feat_dim = 384 + 3 # 3 is in order to concatenate the input images
        feat_dims_pre_agg = [1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 640, 320, 320, 320]
        config, diffusion_extractor, aggregation_network = load_models(config_path, "cuda", batch_size = batch_size, pretrained=False)
        self.config = config
        self.diffusion_extractor = diffusion_extractor
        self.aggregation_network = aggregation_network
        

        # define layer name (for logging)
        self.layer = "all-layers"
        self.checkpoint_name = model_id + "all-timesteps" # f"_noise-{'-'.join([str(_x) for _x in self.timesteps])}"

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
        # image_height, image_width = images.shape[2] // self.patch_size, images.shape[3] // self.patch_size
        image_height, image_width = images.shape[2], images.shape[3]
        with torch.no_grad():
            with torch.autocast("cuda"):
                feats, _ = self.diffusion_extractor.forward(images)
                b, s, l, w, h = feats.shape
        print("model output shape:", b, s, l, w, h)
        spatial = self.aggregation_network(
            feats.float().view((b, -1, w, h))
        )
        # resize to original image size
        spatial = interpolate(spatial, size=(image_height, image_width), mode="bilinear", align_corners=False)
        
        # concatenate with input images
        spatial = torch.cat([images, spatial], dim=1)
        return spatial