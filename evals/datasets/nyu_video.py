"""
MIT License

Copyright (c) 2024 Mohamed El Banani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os
import pickle

import numpy as np
import scipy
import torch

from .utils import get_nyu_transforms


def NYU_video(
    train_path,
    split,
    image_mean="imagenet",
    center_crop=False,
    rotateflip=False,
    augment_train=False,
    frame_count = 16,   
):
    assert split in ["train", "trainval", "valid", "test"]
    return NYU_geonet_video(
        path = train_path,
        split = split,
        center_crop = center_crop,
        augment_train = augment_train,
        image_mean = image_mean,
        rotateflip=rotateflip,
        frame_count = frame_count
    )
   


class NYU_geonet_video(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        split,
        image_mean="imagenet",
        center_crop=False,
        augment_train=False,
        rotateflip=False,
        frame_count = 16,
    ):
        super().__init__()
        
        self.frame_count = frame_count
        self.data = []
        
        self.name = "NYUv2"
        self.center_crop = center_crop
        self.max_depth = 10.0

        # get transforms
        image_size = (480, 480) if center_crop else (480, 640)
        augment = augment_train and "train" in split
        self.image_transform, self.shared_transform = get_nyu_transforms(
            image_mean,
            image_size,
            augment,
            rotateflip=rotateflip,
            additional_targets={"depth": "image", "snorm": "image"},
        )

        # parse dataset
        self.root_dir = path
        insts = os.listdir(path)
        insts.sort()

        

        # remove bad indices
        try: 
            del insts[21181]
        except: 
            print("unable to remove index 21181")   
        try: 
            del insts[6919]
        except: 
            print("unable to remove index 6919")


        # Collect all scenes and their respective files
        scene_files = {}
        for f in insts:
            if f.endswith(".mat"):  # Ensure we're only processing the right files
                # Extract the scene name part, up to the timestamp
                scene_name = f.split('-', 1)[0]  # This splits at the last '-' and takes the first part
                if scene_name not in scene_files:
                    scene_files[scene_name] = []
                scene_files[scene_name].append(f)

        print("number of scenes:", len(scene_files))


        scene_keys = list(scene_files.keys())
        # randomize with seed
        np.random.seed(0)
        np.random.shuffle(scene_keys)

        if split == "train":
            scene_keys = scene_keys[:int(0.8 * len(scene_keys))]
        elif split == "valid":
            scene_keys = scene_keys[int(0.8 * len(scene_keys)):]
        elif split != "trainval":
            raise ValueError()


        # Sort files in each scene by timestamp and prepare data groups
        for scene in scene_keys:
            timestamp = lambda x: float(x.split('-')[1].rsplit('.', 1)[0])
            files = scene_files[scene]
            if len(files) < frame_count:
                continue

            files.sort(key=timestamp)

            # Bundle files into sequences
            for i in range(len(files) - frame_count + 1):
                self.data.append([os.path.join(path, fname) for fname in files[i:i + frame_count]])

        print(f"NYU-GeoNet {split}: {len(self.data)} sets of frames.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_names = self.data[index]
        images = []
        depths = []
        snorms = []
        
        # instances = [scipy.io.loadmat(file_name) for file_name in file_names]
        
        # return instances 
        for file_name in file_names:
            room = "_".join(file_name.split("-")[0].split("_")[:-2])

            # extract elements from the matlab thing
            instance = scipy.io.loadmat(os.path.join(self.root_dir, file_name))
            image = instance["img"][:480, :640]
            depth = instance["depth"][:480, :640]
            snorm = torch.tensor(instance["norm"][:480, :640]).permute(2, 0, 1)

            # process image
            image[:, :, 0] = image[:, :, 0] + 2 * 122.175
            image[:, :, 1] = image[:, :, 1] + 2 * 116.169
            image[:, :, 2] = image[:, :, 2] + 2 * 103.508
            image = image.astype(np.uint8)
            image = self.image_transform(image)

            # set max depth to 10
            # depth[depth > self.max_depth] = 0

            # center crop
            if self.center_crop:
                image = image[..., 80:-80]
                depth = depth[..., 80:-80]
                snorm = snorm[..., 80:-80]

            if self.shared_transform:
                # put in correct format (h, w, feat)
                image = image.permute(1, 2, 0).numpy()
                snorm = snorm.permute(1, 2, 0).numpy()
                depth = depth[:, :, None]

                # transform
                transformed = self.shared_transform(image=image, depth=depth, snorm=snorm)

                # get back in (feat_dim x height x width)
                image = torch.tensor(transformed["image"]).float().permute(2, 0, 1)
                snorm = torch.tensor(transformed["snorm"]).float().permute(2, 0, 1)
                depth = torch.tensor(transformed["depth"]).float()[None, :, :, 0]
            else:
                # move to torch tensors
                depth = torch.tensor(depth).float()[None, :, :]
                snorm = torch.tensor(snorm).float()
            
            images.append(image)
            depths.append(depth)
            snorms.append(snorm)

        images = torch.stack(images)
        depths = torch.stack(depths)
        snorms = torch.stack(snorms)

        return {"image": images, "depth": depths, "snorm": snorms, "room": room}
