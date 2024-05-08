import argparse
import os
import time
from pathlib import Path

import gin
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from core.raft import RAFT
from datasets import get_test_data_loader
from utils.data_utils import crop_operation, scale_operation
from utils.frame_utils import write_pfm, write_vis


@gin.configurable()
def inference(
        test_loader,
        ckpt,
        output_folder,
        rescale=1,
        crop=None,
        do_report=False,
        write_min_depth=None,
    ):

    model = RAFT(test_mode=True).cuda()

    if ckpt is not None:
        tmp = torch.load(ckpt)
        if list(tmp.keys())[0][:7] == "module.":
            model = nn.DataParallel(model)
        model.load_state_dict(tmp, strict=True)
    model.eval()
    
    output_folder = Path(output_folder)
    (output_folder / "depths").mkdir(exist_ok=True, parents=True)

    with torch.no_grad():
        for images, poses, intrinsics, image_names, scale in tqdm(test_loader):
            poses = poses.cuda()
            images = images.squeeze(0)
            intrinsics = intrinsics.squeeze(0)
            images, intrinsics = scale_operation(images, intrinsics, rescale)
            if not crop is None:
                crop_h, crop_w = crop
                images, intrinsics = crop_operation(images, intrinsics, crop_h, crop_w)
            images = images.unsqueeze(0).cuda()
            intrinsics = intrinsics.unsqueeze(0).cuda()
            if do_report:
                tic = time.time()
            disp_est = model(images, poses, intrinsics, do_report=do_report, scale=scale)
            #clip the value minus 0
            disp_est[disp_est < 0] = 1e6
            if do_report:
                print(f"per view time: {time.time() - tic}")
            res = disp_est.cpu().numpy()[0, 0]
            im = np.where(res == 0, 0, 1 / res).astype(np.float32)
            res[res > 50] = 0
            write_pfm(output_folder / "depths" / f"{image_names[0][0]}_scale{rescale}_nf{test_loader.dataset.num_frames}.pfm", im)
            write_vis(output_folder / "depths" / f"{image_names[0][0]}_scale{rescale}_nf{test_loader.dataset.num_frames}.png", im)
            # if write_min_depth is not None:
            #     write_min_depth = Path(write_min_depth)
            #     write_min_depth.mkdir(exist_ok=True)
            #     with open(write_min_depth / f"{image_names[0][0]}.txt", "w") as f:
            #         min_depth = np.quantile(im[im > 0], 0.1) / 2
            #         f.write(f"{min_depth}\n")
            torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gin_config', nargs='+', default=['inference_DTU'],
                        help='Set of config files for gin (separated by spaces) '
                        'e.g. --gin_config file1 file2 (exclude .gin from path)')
    parser.add_argument('-p', '--gin_param', nargs='+', default=[],
                        help='Parameter settings that override config defaults '
                        'e.g. --gin_param module_1.a=2 module_2.b=3')
    args = parser.parse_args()
    gin_files = [f'configs/{g}.gin' for g in args.gin_config]
    gin.parse_config_files_and_bindings(
        gin_files, args.gin_param, skip_unknown=True)

    test_loader = get_test_data_loader()
    inference(test_loader)