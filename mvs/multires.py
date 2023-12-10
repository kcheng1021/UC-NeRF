import argparse
import os
import subprocess
from pathlib import Path

import cv2
import gin
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils.frame_utils import readPFM, write_pfm


@gin.configurable()
def multires(output_folder, suffix1="", suffix2="", th=0.02, down_sample=1, visualize=False):
    output_folder = Path(output_folder)
    names = os.listdir(output_folder / "depths")
    names = sorted([name.split("_scale1")[0] for name in names if "_scale1" in name])
    n = len(names)

    for i in tqdm(range(n)):
        output = output_folder / "depths" / f"{names[i]}{suffix1}{suffix2}_th{th}.pfm"
        im1 = readPFM(output_folder / "depths" / f"{names[i]}_scale0.5{suffix1}.pfm")
        im2 = readPFM(output_folder / "depths" / f"{names[i]}_scale1{suffix2}.pfm")
        im1 = cv2.resize(im1, im2.shape[::-1])
        mask = np.abs(im1 - im2) < th * im1
        im = np.where(mask, im2, im1)
        if down_sample != 1:
            im = cv2.resize(im, tuple(np.array(im.shape[::-1]) // down_sample))
        write_pfm(output, im)
        if visualize:
            d = 1 / im
            d[np.isnan(d)] = 0
            d = np.minimum(np.maximum(d, 0), 5 * np.median(d))
            plt.figure(figsize=(20,20))
            plt.imshow(d)
            vis_output = output_folder / "depths" / f"{names[i]}.png"
            plt.savefig(vis_output)
            plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse. May be different from the original implementation')
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

    multires()