import os
from pathlib import Path

import gin
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

from utils.frame_utils import read_gen


@gin.configurable()
class Custom(Dataset):
    def __init__(self, dataset_path, num_frames, min_dist_over_baseline=1, cam_format="TUM", subset=None, window_stride=1, **args):
        self.dataset_path = Path(dataset_path)
        self.image_formats = [".jpg", ".jpeg", ".png"]
        self.data_index = []
        for f in os.listdir(self.dataset_path / "images"):
            if os.path.splitext(f)[-1] in self.image_formats:
                self.image_formats = [os.path.splitext(f)[-1]]
                self.data_index.append(os.path.splitext(f)[0])
        self.data_index = sorted(self.data_index)
        len0 = len(self.data_index)
        if subset is not None:
            self.data_index = [self.data_index[x] for x in subset]
        if cam_format == "TUM":
            self.poses = np.zeros((len(self.data_index), 4, 4), dtype=np.float)
            cams_path = self.dataset_path / "cams.txt"
            poses = np.loadtxt(cams_path, dtype=np.float)[:, 1:]
            assert(len(poses) == len0)
            if subset is not None: poses = poses[subset]
            self.cam_centers = []
            for i in range(len(poses)):
                self.cam_centers.append(poses[i, :3])
                self.poses[i, :3, :3] = R.from_quat(poses[i, 3:]).as_matrix()
                self.poses[i, :3, 3] = self.cam_centers[-1]
                self.poses[i, 3, 3] = 1
                self.poses[i] = np.linalg.inv(self.poses[i])
            intrinsic_path = self.dataset_path / "intrinsic.txt"
            intrinsic = np.loadtxt(intrinsic_path, dtype=np.float)
            self.intrinsics = [intrinsic] * len(poses)
        if min_dist_over_baseline is not None:
            baselines = []
            for i in range(len(self.poses) - 1):
                baselines.append(np.linalg.norm(self.cam_centers[i] - self.cam_centers[i + 1]))
            self.min_depth = np.mean(baselines) * min_dist_over_baseline
        else:
            self.min_depth = None
        self.num_frames = num_frames
        self.image_format = self.image_formats[0]
        self.offsets = np.arange(-num_frames // 2, -num_frames // 2 + num_frames + 1) * window_stride
        self.window_stride = window_stride



    def __len__(self):
        return len(self.data_index)
    
    def __getitem__(self, index):
        indices = self.offsets.copy() + index
        while indices[0] < 0:
            indices += self.window_stride
        while indices[-1] >= len(self.data_index):
            indices -= self.window_stride
        assert(indices[0] >= 0)
        indices = [index] + [i for i in indices if i != index]
        images, poses, intrinsics = [], [], []
        for i in indices:
            image = read_gen(str(self.dataset_path / "images" / f"{self.data_index[i]}{self.image_format}"))
            images.append(image)
            poses.append(self.poses[i])
            intrinsics.append(self.intrinsics[i])

        if self.min_depth is None:
            depth_path = self.dataset_path / "min_depth" / f"{self.data_index[index]}.txt"
            scale_info = np.loadtxt(depth_path, dtype=np.float)
            scale = 400 / scale_info
        else:
            scale = 400 / self.min_depth

        images = np.stack(images, 0).astype(np.float32)
        poses = np.stack(poses, 0).astype(np.float32)
        intrinsics = np.stack(intrinsics, 0).astype(np.float32)

        images = torch.from_numpy(images)
        poses = torch.from_numpy(poses)
        intrinsics = torch.from_numpy(intrinsics)

        # channels first
        images = images.permute(0, 3, 1, 2)
        images = images.contiguous()
        
        return images, poses, intrinsics, [self.data_index[i] for i in indices], scale

