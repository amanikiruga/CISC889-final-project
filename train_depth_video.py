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
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import hydra
import omegaconf
import torch
import torch.multiprocessing as mp
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.distributed import destroy_process_group, init_process_group


from torch.nn.functional import interpolate
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from evals.datasets.builder import build_loader
from evals.utils.losses import DepthLoss
from evals.utils.metrics import evaluate_depth, match_scale_and_shift
from evals.utils.optim import cosine_decay_linear_warmup
import wandb 


def ddp_setup(rank: int, world_size: int, port: int):
    """
    Args:
        rank: Unique identifier of each process
       world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def train(
    model,
    probe,
    train_loader,
    optimizer,
    scheduler,
    n_epochs,
    detach_model,
    loss_fn,
    rank=0,
    world_size=1,
    valid_loader=None,
    scale_invariant=False,
    log_interval=200,
    save_interval = 2000, 
    cfg = None, 
    exp_path = None, 
):
    for ep in range(n_epochs):
        if world_size > 1:
            train_loader.sampler.set_epoch(ep)

        train_loss = 0
        pbar = tqdm(train_loader) if rank == 0 else train_loader
        for i, batch in enumerate(pbar):
            images = batch["image"].to(rank).squeeze(0)
            target = batch["depth"].to(rank).squeeze(0)[:cfg.batch_size]
            

            optimizer.zero_grad()
            if detach_model:
                with torch.no_grad():
                    feats = model(images)
                    if isinstance(feats, (tuple, list)):
                        feats = [_f.detach() for _f in feats]
                    else:
                        feats = feats.detach()
            else:
                feats = model(images)
            pred = probe(feats)[:cfg.batch_size]
            print("pred stats", pred.min(), pred.max())
            print("target stats", target.min(), target.max())
            pred = interpolate(pred, size=target.shape[-2:], mode="bilinear")

            if scale_invariant:
                pred = match_scale_and_shift(pred, target)
                pred = pred.clamp(min=0.001, max=10.0)
            loss = loss_fn(pred, target)
            print("pred stats", pred.min(), pred.max()) 
            print("target stats", target.min(), target.max() )
            loss.backward()
            optimizer.step()
            scheduler.step()

            pr_lr = optimizer.param_groups[0]["lr"]
            loss = loss.item()
            train_loss += loss

            if rank == 0:
                _loss = train_loss / (i + 1)
                pbar.set_description(
                    f"{ep} | loss: {loss:.4f} ({_loss:.4f}) probe_lr: {pr_lr:.2e}"
                )
                if i % log_interval == 0:
                    wandb.log({"loss": _loss, "probe_lr": pr_lr})
                    # log prediction and target 
                    wandb.log({"prediction": wandb.Image(pred[:8].cpu().detach())})
                    wandb.log({"target": wandb.Image(target[:8].cpu().detach())})

            if i % save_interval == 0 and rank == 0:
                ckpt_path = exp_path / f"ckpt_{ep}_{i}.pth"
                checkpoint = {
                    "cfg": cfg,
                    "probe": probe.state_dict(),
                }
                torch.save(checkpoint, ckpt_path)
                logger.info(f"Saved checkpoint at {ckpt_path}")


        train_loss /= len(train_loader)

        if rank == 0:
            logger.info(f"train loss {ep}   | {train_loss:.4f}")
            if valid_loader is not None:
                val_loss, val_metrics = validate(
                    model, probe, valid_loader, loss_fn, scale_invariant=scale_invariant
                )
                logger.info(f"valid loss {ep}   | {val_loss:.4f}")
                for metric in val_metrics:
                    logger.info(f"valid SA {metric:10s} | {val_metrics[metric]:.4f}")


def validate(
    model, probe, loader, loss_fn, verbose=True, scale_invariant=False, aggregate=True, cfg = None, log_interval = 20
):
    total_loss = 0.0
    metrics = None
    with torch.inference_mode():
        pbar = tqdm(loader, desc="Evaluation") if verbose else loader
        for i, batch in enumerate(pbar):
            images = batch["image"].cuda().squeeze(0)
            target = batch["depth"].cuda().squeeze(0)[:cfg.batch_size]

            feat = model(images)
            pred = probe(feat).detach()[:cfg.batch_size]
            pred = interpolate(pred, size=target.shape[-2:], mode="bilinear")

            if i % log_interval == 0:
                wandb.log({"val_prediction": wandb.Image(pred[:8].cpu().detach())})
                wandb.log({"val_target": wandb.Image(target[:8].cpu().detach())})

            loss = loss_fn(pred, target)
            total_loss += loss.item()

            batch_metrics = evaluate_depth(
                pred, target, scale_invariant=scale_invariant
            )
            if metrics is None:
                metrics = {
                    key: [
                        value,
                    ]
                    for key, value in batch_metrics.items()
                }
            else:
                for key, value in batch_metrics.items():
                    metrics[key].append(value)

    # aggregate
    total_loss = total_loss / len(loader)


    for key in metrics:
        metric_key = torch.cat(metrics[key], dim=0)
        metrics[key] = metric_key.mean() if aggregate else metric_key

    return total_loss, metrics


def train_model(rank, world_size, cfg):
    if world_size > 1:
        ddp_setup(rank, world_size, cfg.system.port)

    # ===== GET DATA LOADERS =====
    # validate and test on single gpu
    # train_loader = build_loader(cfg.dataset, "train", 1, world_size)
    # test_loader = build_loader(cfg.dataset, "valid", 1, 1)
    train_loader = build_loader(cfg.dataset, "trainval", cfg.batch_size, world_size)
    test_loader = build_loader(cfg.dataset, "test", cfg.batch_size, 1)
    train_loader.dataset.__getitem__(0)

    # ===== Get models =====
    model = instantiate(cfg.backbone)
    probe = instantiate(
        cfg.probe, feat_dim=model.feat_dim, max_depth=train_loader.dataset.max_depth
    )

    # setup experiment name
    # === job info
    timestamp = datetime.now().strftime("%d%m%Y-%H%M")
    train_dset = train_loader.dataset.name
    test_dset = test_loader.dataset.name
    model_info = [
        f"{model.checkpoint_name:40s}",
        f"{model.patch_size:2d}",
        f"{str(model.layer):5s}",
        f"{model.output:10s}",
    ]
    probe_info = [f"{probe.name:25s}"]
    batch_size = cfg.batch_size * cfg.system.num_gpus
    train_info = [
        f"{cfg.optimizer.n_epochs:3d}",
        f"{cfg.optimizer.warmup_epochs:4.2f}",
        f"{str(cfg.optimizer.probe_lr):>10s}",
        f"{str(cfg.optimizer.model_lr):>10s}",
        f"{batch_size:4d}",
        f"{train_dset:10s}",
        f"{test_dset:10s}",
    ]

    # define exp_name
    exp_name = "_".join([timestamp] + model_info + probe_info + train_info)
    exp_name = f"{exp_name}_{cfg.note}" if cfg.note != "" else exp_name
    exp_name = exp_name.replace(" ", "")  # remove spaces

    # ===== SETUP LOGGING =====
    if rank == 0:
        exp_path = Path(__file__).parent / f"depth_exps/{exp_name}"
        exp_path.mkdir(parents=True, exist_ok=True)
        logger.add(exp_path / "training.log")
        logger.info(f"Config: \n {OmegaConf.to_yaml(cfg)}")

    # move to cuda
    model = model.to(rank)
    probe = probe.to(rank)

    # very hacky ... SAM gets some issues with DDP finetuning
    model_name = model.checkpoint_name
    if "sam" in model_name or "vit-mae" in model_name:
        h, w = train_loader.dataset.__getitem__(0)["image"].shape[-2:]
        model.resize_pos_embed(image_size=(h, w))

    # move to DDP
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        probe = DDP(probe, device_ids=[rank])

    if cfg.optimizer.model_lr == 0:
        optimizer = torch.optim.AdamW(
            [{"params": probe.parameters(), "lr": cfg.optimizer.probe_lr}]
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": probe.parameters(), "lr": cfg.optimizer.probe_lr},
                {"params": model.parameters(), "lr": cfg.optimizer.model_lr},
            ]
        )

    # lambda_fn = lambda epoch: cosine_decay_linear_warmup(  # noqa: E731
    #     epoch,
    #     cfg.optimizer.n_epochs * len(train_loader),
    #     cfg.optimizer.warmup_epochs * len(train_loader),
    # )
    lambda_fn = lambda epoch: 1.0
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_fn)
    loss_fn = instantiate(cfg.loss)

    train(
        model,
        probe,
        train_loader,
        optimizer,
        scheduler,
        cfg.optimizer.n_epochs,
        detach_model=(cfg.optimizer.model_lr == 0),
        loss_fn=loss_fn,
        rank=rank,
        world_size=world_size,
        cfg = cfg, 
        exp_path = exp_path,    
        log_interval=cfg.log_interval or 200,
        # scale_invariant = cfg.scale_invariant,
        # valid_loader=test_loader,     
    )

    if rank == 0:
        logger.info(f"Evaluating on test split of {test_dset}")

        test_sa_loss, test_sa_metrics = validate(model, probe, test_loader, loss_fn, cfg = cfg)
        logger.info(f"Scale-Aware Final test loss       | {test_sa_loss:.4f}")
        for metric in test_sa_metrics:
            logger.info(f"Final test SA {metric:10s} | {test_sa_metrics[metric]:.4f}")

        wandb.log({"test_loss": test_sa_loss})
        for metric in test_sa_metrics:
            wandb.log({"test_sa_" + metric: test_sa_metrics[metric]})
        results_sa = ", ".join([f"{test_sa_metrics[_m]:.4f}" for _m in test_sa_metrics])

        # get scale invariant
        test_si_loss, test_si_metrics = validate(
            model, probe, test_loader, loss_fn, scale_invariant=True, cfg = cfg
        )
        logger.info(f"Scale-Invariant Final test loss       | {test_si_loss:.4f}")
        for metric in test_si_metrics:
            logger.info(f"Final test SI {metric:10s} | {test_si_metrics[metric]:.4f}")
        
        wandb.log({"test_si_loss": test_si_loss})
        for metric in test_si_metrics:
            wandb.log({"test_si_" + metric: test_si_metrics[metric]})

        results_si = ", ".join([f"{test_si_metrics[_m]:.4f}" for _m in test_si_metrics])

        # log experiments
        exp_info = ", ".join(model_info + probe_info + train_info)
        log = f"{timestamp}, {exp_info}, {results_sa}, {results_si} \n"
        with open(f"depth_results_{test_dset}.log", "a") as f:
            f.write(log)

        # save final model
        ckpt_path = exp_path / "ckpt.pth"
        checkpoint = {
            "cfg": cfg,
            "probe": probe.state_dict(),
        }
        torch.save(checkpoint, ckpt_path)
        logger.info(f"Saved checkpoint at {ckpt_path}")

    if world_size > 1:
        destroy_process_group()


@hydra.main(config_name="depth_video_training", config_path="./configs", version_base=None)
def main(cfg: DictConfig):
    # initialize wandb 
    wandb.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(project="depth_video_probe", name=cfg.note)
    world_size = cfg.system.num_gpus
    if world_size > 1:
        mp.spawn(train_model, args=(world_size, cfg), nprocs=world_size)
    else:
        train_model(0, world_size, cfg)


if __name__ == "__main__":
    main()
