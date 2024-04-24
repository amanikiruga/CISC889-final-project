import os
import random
import torch
import einops
import math
import argparse
import time

from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import json
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

def denormalize(tensor, mean, std):
    """
    Denormalizes a tensor that was normalized with the given mean and std.
    """
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # The inverse of normalization
    return tensor

def calculate_absrel(true_depth, predicted_depth):
    """Calculate the absolute relative difference with renormalization and masking for zero true depth values.

    Args:
        true_depth (torch.Tensor): Ground truth depth values in the range -1 to 1.
        predicted_depth (torch.Tensor): Predicted depth values in the range -1 to 1.

    Returns:
        float: Absolute relative difference, excluding zero depth values.
    """
    # Ensure the ground truth and predicted depths have the same shape
    if true_depth.shape != predicted_depth.shape:
        raise ValueError(
            "The shapes of true_depth and predicted_depth must be the same."
        )

    # Renormalize depths from -1, 1 to 0, 1
    true_depth = (true_depth + 1) / 2
    predicted_depth = (predicted_depth + 1) / 2

    # Create a mask for non-zero true depth values
    mask = true_depth != 0

    # Apply the mask
    true_depth_masked = true_depth[mask]
    predicted_depth_masked = predicted_depth[mask]

    # Calculate absolute relative difference
    abs_rel = torch.mean(
        torch.abs(true_depth_masked - predicted_depth_masked) / true_depth_masked
    )

    return abs_rel.item()


def calculate_delta1(true_depth, predicted_depth, threshold=1.25):
    """Calculate the percentage of pixels where predicted depth is within a threshold."""
    max_ratio = torch.maximum(
        predicted_depth / true_depth, true_depth / predicted_depth
    )
    delta1 = (max_ratio < threshold).float().mean()
    return delta1.item()


def show_images(data_loader, title):
    # Fetch one batch of images
    images, depths = next(iter(data_loader))

    # Denormalize and prepare the images for display
    images = [
        denormalize(img, mean, std).numpy().transpose((1, 2, 0)) for img in images
    ]
    depths = [d.numpy() for d in depths.squeeze(1)]

    # Plotting
    plt.figure(figsize=(6, 5))
    for i in range(len(images)):
        plt.subplot(2, len(images), i + 1)
        plt.imshow(images[i])
        plt.title(f"{title} RGB")
        plt.axis("off")

        plt.subplot(2, len(images), len(images) + i + 1)
        plt.imshow(depths[i], cmap="gray")
        plt.title(f"{title} Depth")
        plt.axis("off")

    plt.show()


class SpatiallyAlignedHead(nn.Module):
    def __init__(self, input_channels, output_channels=1):
        super(SpatiallyAlignedHead, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=3, padding=1)
        self.silu = nn.SiLU()
        self.conv2 = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, output_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.silu(self.conv1(x))
        x = self.silu(self.conv2(x))
        x = self.tanh(self.conv3(x))
        return x


class ReDWebDataset(Dataset):
    """
    all_file_names = [
        f.split(".")[0]
        for f in os.listdir(
            os.path.join("/lustre/scratch/diff/datasets/ReDWEBv1/ReDWeb_V1", "Imgs")
        )
    ]

    # Shuffle and split the file names
    random.shuffle(all_file_names)
    num_test_samples = 600
    test_file_names = all_file_names[:num_test_samples]
    train_file_names = all_file_names[num_test_samples:]

    # Create Dataset instances for train and test
    train_dataset = ReDWebDataset(
        root_dir="/lustre/scratch/diff/datasets/ReDWEBv1/ReDWeb_V1",
        file_names=train_file_names,
        image_transform=image_transform,
        depth_transform=depth_transform,
    )
    test_dataset = ReDWebDataset(
        root_dir="/lustre/scratch/diff/datasets/ReDWEBv1/ReDWeb_V1",
        file_names=test_file_names,
        image_transform=image_transform,
        depth_transform=depth_transform,
    )

    # Create Data Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last = True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=True, drop_last = True
    )
    """

    def __init__(
        self, root_dir, file_names, image_transform=None, depth_transform=None
    ):
        self.root_dir = root_dir
        self.file_names = file_names
        self.image_transform = image_transform
        self.depth_transform = depth_transform
        self.img_dir = os.path.join(root_dir, "Imgs")
        self.depth_dir = os.path.join(root_dir, "RDs")

    # rest of the class remains the same

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.file_names[idx] + ".jpg")
        depth_name = os.path.join(self.depth_dir, self.file_names[idx] + ".png")

        # Load image and depth map
        image = Image.open(img_name).convert("RGB")
        depth = Image.open(depth_name).convert("L")  # Convert depth to grayscale

        if self.image_transform:
            image = self.image_transform(image)
        if self.depth_transform:
            depth = self.depth_transform(depth)

        return image, depth


def evaluate_test_set(
    test_loader, model, diffusion_extractor, aggregation_network, device
):
    model.eval()  # Set the model to evaluation mode
    absrel_sum = 0.0
    delta1_sum = 0.0
    count = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.autocast("cuda"):
                feats, _ = diffusion_extractor.forward(inputs)
                b, s, l, w, h = feats.shape
                diffusion_hyperfeats = aggregation_network(
                    feats.float().view((b, -1, w, h))
                )
            outputs = model(diffusion_hyperfeats)
            absrel_sum += calculate_absrel(labels, outputs)
            delta1_sum += calculate_delta1(labels, outputs)
            count += 1

    model.train()  # Set the model back to training mode
    return absrel_sum / count, delta1_sum / count


class VOC2010Dataset(Dataset):
    def __init__(
        self, root_dir, file_names, image_transform=None, depth_transform=None
    ):
        self.root_dir = root_dir
        self.file_names = file_names
        self.image_transform = image_transform
        self.depth_transform = depth_transform
        self.img_dir = os.path.join(root_dir, "JPEGImages")
        self.depth_dir = os.path.join(root_dir, "DepthImages")

    # rest of the class remains the same

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.file_names[idx] + ".jpg")
        depth_name = os.path.join(
            self.depth_dir, self.file_names[idx] + ".jpg_depth.pt"
        )

        # Load image and depth map
        image = Image.open(img_name).convert("RGB")
        depth = torch.load(depth_name)

        depth = depth.unsqueeze(0)

        # normalize depth to -1 to 1 because of tanh activation
        depth = depth / (depth.max() - depth.min()) * 2 - 1

        if self.image_transform:
            image = self.image_transform(image)
        if self.depth_transform:
            depth = self.depth_transform(depth)

        return image, depth


def parse_arguments():
    parser = argparse.ArgumentParser(description="Description of your script.")
    # Add arguments
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    # store true type of argument
    parser.add_argument(
        "--train_feature_extractor",
        action="store_true",
        default=False,
        help="Whether to train feature extractor",
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="diffusion-hyperfeatures",
        help="Name of the experiment",
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints",
        help="Path to the checkpoint",
    )

    parser.add_argument(
        "--train_feature_extractor_from_scratch",
        action="store_true",
        default=False,
        help="Whether to train feature extractor from scratch",
    )

    parser.add_argument(
        "--test_only",
        action="store_true",
        default=False,
        help="Whether to test only",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    wandb.init(project="readout-depth", name=args.experiment_name, config=args)
    device = args.device

    # Memory requirement is 13731MiB
    if args.train_feature_extractor_from_scratch:
        print("Using train scratch config")
        config_path = (
            "configs/train.yaml"  # load model from scratch instead of pretrained
        )
    else:  # load pretrained model
        print("Using pretrained config")
        config_path = "configs/real.yaml"
    config, diffusion_extractor, aggregation_network = load_models(config_path, device)

    # Transformations
    image_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # Normalize using ImageNet stats for RGB images
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    depth_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
        ]
    )

    # prepare the dataset

    # Get the complete list of file names (without extension)
    root_dir = "/lustre/scratch/diff/datasets/VOC2010/VOCdevkit/VOC2010"

    all_file_names = [
        f.split(".")[0] for f in os.listdir(os.path.join(root_dir, "JPEGImages"))
    ]

    # Shuffle and split the file names
    random.shuffle(all_file_names)
    num_test_samples = 1000
    test_file_names = all_file_names[:num_test_samples]
    train_file_names = all_file_names[num_test_samples:]

    # Create Dataset instances for train and test
    train_dataset = VOC2010Dataset(
        root_dir=root_dir,
        file_names=train_file_names,
        image_transform=image_transform,
        depth_transform=depth_transform,
    )
    test_dataset = VOC2010Dataset(
        root_dir=root_dir,
        file_names=test_file_names,
        image_transform=image_transform,
        depth_transform=depth_transform,
    )

    # Create Data Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
    )

    # training params
    input_channels = 384

    readout_head = SpatiallyAlignedHead(input_channels=input_channels).to(device)

    criterion = nn.MSELoss()

    if args.train_feature_extractor:
        parameter_groups = [
            {"params": aggregation_network.mixing_weights, "lr": config["lr"]},
            {
                "params": aggregation_network.bottleneck_layers.parameters(),
                "lr": config["lr"],
            },
            {
                "params": readout_head.parameters(),
                "lr": config["lr"],
            },
        ]
    else:
        parameter_groups = [
            {
                "params": readout_head.parameters(),
                "lr": 0.001,
            },
        ]

    optimizer = optim.AdamW(parameter_groups, lr=1e-3)  # Adjust learning rate as needed

    num_epochs = 10  # Set the number of epochs
    checkpoint_dir = os.path.join(args.checkpoint_path, args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    latest_checkpoint_path = None
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
        if checkpoint_files:
            # Sort based on the epoch number
            checkpoint_files.sort(
                key=lambda x: int(x.split("_")[-1].split(".")[0]), reverse=True
            )
            latest_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])

    # Load the latest checkpoint if it exists
    if latest_checkpoint_path is not None:
        print(f"Resuming training from the latest checkpoint: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path)
        readout_head.load_state_dict(checkpoint["readout_state_dict"])
        aggregation_network.load_state_dict(checkpoint["aggregation_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0

    # Assuming you've defined the SpatiallyAlignedHead and dataloaders

    # Training Loop with tqdm and wandb
    num_epochs = 10
    start_time = time.time()  # Start timer

    if args.test_only:
        absrel_metric, delta1_metric = evaluate_test_set(
            test_loader,
            readout_head,
            diffusion_extractor,
            aggregation_network,
            device,
        )
        print(f"Test AbsRel: {absrel_metric}, Test Delta1: {delta1_metric}")
        wandb.log(
            {
                "Test set AbsRel": absrel_metric,
                "Test set Delta1": delta1_metric,
            }
        )
        print("Test set evaluation complete")
        return
    for epoch in range(num_epochs):
        # running_loss = 0.0

        # Training loop with tqdm progress bar
        for i, data in enumerate(tqdm(train_loader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            print(inputs.shape)
            # Forward pass through your diffusion extractor and aggregation network
            if args.train_feature_extractor:
                with torch.no_grad():
                    with torch.autocast("cuda"):
                        feats, _ = diffusion_extractor.forward(inputs)
                        b, s, l, w, h = feats.shape
                diffusion_hyperfeats = aggregation_network(
                    feats.float().view((b, -1, w, h))
                )
            else:
                with torch.no_grad():
                    with torch.autocast("cuda"):
                        feats, _ = diffusion_extractor.forward(inputs)
                        b, s, l, w, h = feats.shape
                        diffusion_hyperfeats = aggregation_network(
                            feats.float().view((b, -1, w, h))
                        )

            # Forward pass through the readout head
            outputs = readout_head(diffusion_hyperfeats)

            # Calculate loss and backpropagate
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            end_time = time.time()  # End timer
            iteration_time = end_time - start_time  # Calculate iteration duration

            # Logging to wandb
            wandb.log(
                {
                    "Loss": loss.item(),
                    "Iteration": epoch * len(train_loader) + i,
                    "Time": iteration_time,
                }
            )

            # Optionally, log images and predictions after certain iterations
            if i % 200 == 0:
                test_inputs, test_labels = next(iter(test_loader))
                test_inputs, test_labels = test_inputs.to(device), test_labels.to(
                    device
                )
                # Forward pass through your diffusion extractor and aggregation network
                with torch.no_grad():
                    with torch.autocast("cuda"):
                        feats, _ = diffusion_extractor.forward(test_inputs)
                        b, s, l, w, h = feats.shape
                        diffusion_hyperfeats = aggregation_network(
                            feats.float().view((b, -1, w, h))
                        )

                wandb.log(
                    {
                        "Train Images": [wandb.Image(img) for img in inputs],
                        "Train Labels": [wandb.Image(label) for label in labels],
                        "Train Predictions": [wandb.Image(pred) for pred in outputs],
                    }
                )

                # Evaluate on test data and log test images and predictions
                with torch.no_grad():
                    test_outputs = readout_head(diffusion_hyperfeats)

                # Calculate metrics
                absrel_metric = calculate_absrel(test_labels, test_outputs)
                delta1_metric = calculate_delta1(test_labels, test_outputs)

                # Log metrics along with images and predictions
                wandb.log(
                    {
                        "Test Images": [wandb.Image(img) for img in test_inputs],
                        "Test Predictions": [
                            wandb.Image(pred) for pred in test_outputs
                        ],
                        "Test Labels": [wandb.Image(label) for label in test_labels],
                        "Test sample AbsRel": absrel_metric,
                        "Test sample Delta1": delta1_metric,
                    }
                )

                if args.train_feature_extractor:
                    # Collect mean gradients for aggregation network
                    aggregation_network_gradients = [
                        param.grad.abs().mean().item()
                        for param in aggregation_network.parameters()
                        if param.grad is not None
                    ]

                    # Collect mean gradients for readout head
                    readout_head_gradients = [
                        param.grad.abs().mean().item()
                        for param in readout_head.parameters()
                        if param.grad is not None
                    ]

                    # Collect mean gradients for mixing weights
                    mixing_weights_gradients = aggregation_network.mixing_weights.grad
                    # Compute the mean of means for the gradients of the aggregation network
                    aggregation_network_mean_grad = np.mean(
                        aggregation_network_gradients
                    )
                    wandb.log(
                        {
                            "Mean Aggregation Network Gradient": aggregation_network_mean_grad
                        }
                    )
                    print("Just logged mean aggregation network gradient")

                    # Compute the mean of means for the gradients of the readout head
                    readout_head_mean_grad = np.mean(readout_head_gradients)
                    wandb.log({"Mean Readout Head Gradient": readout_head_mean_grad})
                    print("Just logged mean readout head gradient")

                    # Compute the mean of means for the gradients of the mixing weights
                    mixing_weights_mean_grad = np.mean(
                        mixing_weights_gradients.detach().cpu().numpy()
                    )
                    wandb.log(
                        {"Mean Mixing Weights Gradient": mixing_weights_mean_grad}
                    )

                    # log histogram of mixing weights
                    wandb.log(
                        {
                            "Mixing Weights Histogram": wandb.Histogram(
                                aggregation_network.mixing_weights.detach()
                                .cpu()
                                .numpy()
                            )
                        }
                    )
                    print("Just logged mean mixing weights gradient")

            if i % 1000 == 0:
                absrel_metric, delta1_metric = evaluate_test_set(
                    test_loader,
                    readout_head,
                    diffusion_extractor,
                    aggregation_network,
                    device,
                )
                print(
                    f"Epoch {epoch}: Test AbsRel: {absrel_metric}, Test Delta1: {delta1_metric}"
                )
                wandb.log(
                    {
                        "Epoch": epoch,
                        "Test set AbsRel": absrel_metric,
                        "Test set Delta1": delta1_metric,
                    }
                )

        checkpoint_path = os.path.join(
            checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"
        )
        torch.save(
            {
                "epoch": epoch,
                "readout_state_dict": readout_head.state_dict(),
                "aggregation_state_dict": aggregation_network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            checkpoint_path,
        )

        print(f"Checkpoint saved for epoch {epoch+1}")

        print(f"Epoch {epoch + 1} completed")

    wandb.finish()


if __name__ == "__main__":
    main()
