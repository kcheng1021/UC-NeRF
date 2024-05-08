import argparse
import os
import time

import gin
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler

from core.raft import RAFT
from datasets import get_train_data_loader
from loss import sequence_loss
from utils.logger import Logger


@gin.configurable('optimizer')
def fetch_optimizer(model, wdecay=.00005, epsilon=1e-8, pct_start=0.001, lr=0.00025, num_steps=None):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wdecay, eps=epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, num_steps+100,
        pct_start=pct_start, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    
def count_parameters(model):
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"{n} {p.numel()}")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



@gin.configurable()
def train(name='test',
        overlap=False,
        batch_size=2,
        SAVE_FREQ=5000,
        fix_gradual_weight=None,
        num_steps=100000
    ):

    model = RAFT().cuda()
    model.train()
    # print(count_parameters(model))

    optimizer, scheduler = fetch_optimizer(model, num_steps=num_steps)

    train_loader = get_train_data_loader(batch_size=batch_size)
    total_steps = 0
    scaler = GradScaler(enabled=True)
    model = nn.DataParallel(model)
    model.name = name
    logger = Logger(model, scheduler)

    tic = None
    total_time = 0
    should_keep_training = True

    initial_steps = total_steps

    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):

            optimizer.zero_grad()
            images, depths, poses, intrinsics = data_blob
            depths = depths.cuda()
            depths = depths[:, [0]]
            disp_gt = torch.where(depths>0, 1.0/depths, torch.zeros_like(depths))

            disp_est = model(images.cuda(), poses.cuda(), intrinsics.cuda())

            if not fix_gradual_weight is None:
                gradual_weight = fix_gradual_weight
            else:
                gradual_weight = total_steps * 1.0 / num_steps

            loss, metrics = sequence_loss(disp_est, disp_gt, gradual_weight=gradual_weight)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            # check for scaler scale if it's vanishing
            print("scaler.get_scale()", scaler.get_scale())
            print("metrics", metrics)

            logger.push(metrics)
            if total_steps % SAVE_FREQ == SAVE_FREQ - 1 or total_steps == 1 or total_steps == num_steps:
                if not overlap and total_steps + 1 != num_steps:
                    PATH = f'checkpoints/{total_steps+1}_{name}.pth'
                else:
                    PATH = f'checkpoints/{name}.pth'
                checkpoint = model.state_dict()
                torch.save(checkpoint, PATH)

            total_steps += 1

            if not tic is None:
                total_time += time.time() - tic
                print(f"time per step: {total_time / (total_steps - initial_steps - 1)}, expected: {total_time / (total_steps - 1 - initial_steps) * (num_steps - initial_steps) / 24 / 3600} days")
            tic = time.time()
            if total_steps > num_steps:
                should_keep_training = False
                break

    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1234)

    parser.add_argument('-g', '--gin_config', nargs='+', default=[],
                        help='Set of config files for gin (separated by spaces) '
                        'e.g. --gin_config file1 file2 (exclude .gin from path)')
    parser.add_argument('-p', '--gin_param', nargs='+', default=[],
                        help='Parameter settings that override config defaults '
                        'e.g. --gin_param module_1.a=2 module_2.b=3')

    args = parser.parse_args()
    

    gin_files = [f'configs/{g}.gin' for g in args.gin_config]
    gin.parse_config_files_and_bindings(
        gin_files, args.gin_param, skip_unknown=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # torch.set_deterministic(True)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train()