#!/usr/bin/env python
import math
import logging
from pathlib import Path
from pprint import pformat
import time

import torch
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.utils import cycle
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.train_utils import get_safe_torch_device
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.logging_utils import AverageMeter
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig


@parser.wrap()
def lr_finder(cfg: TrainPipelineConfig):
    cfg.validate()

    logging.info(pformat(cfg.to_dict()))
    set_seed(cfg.seed)

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True

    logging.info("Creating dataset and policy")
    dataset = make_dataset(cfg)
    policy = make_policy(cfg.policy, ds_meta=dataset.meta).to(device)
    optimizer, _ = make_optimizer_and_scheduler(cfg, policy)
    scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    dl_iter = cycle(dataloader)

    lr_start = 1e-7
    lr_end = 10
    num_steps = 100
    lr_mult = (lr_end / lr_start) ** (1 / num_steps)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_start

    policy.train()
    losses = []
    lrs = []

    logging.info("Starting LR Finder")
    for step in range(num_steps):
        batch = next(dl_iter)
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type) if cfg.policy.use_amp else torch.no_grad():
            loss, _ = policy.forward(batch)

        if torch.isnan(loss) or torch.isinf(loss):
            break

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(policy.parameters(), cfg.optimizer.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        lr = optimizer.param_groups[0]["lr"]
        losses.append(loss.item())
        lrs.append(lr)

        for param_group in optimizer.param_groups:
            param_group["lr"] *= lr_mult

    logging.info("Plotting LR Finder curve")
    plt.figure()
    plt.plot(lrs, losses)
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("LR Finder")
    plt.grid(True)
    plt.savefig(Path(cfg.output_dir) / "lr_finder.png")
    plt.show()

    logging.info("Done with LR Finder")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    lr_finder()
