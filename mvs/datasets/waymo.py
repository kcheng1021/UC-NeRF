import os
from pathlib import Path

import gin
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

from utils.frame_utils import read_gen
import cv2
import pickle
import json
from PIL import Image

@gin.configurable()
class Waymo(Dataset):
    def __init__(self, dataset_path, num_frames, min_dist_over_baseline=1, cam_format="TUM", subset=None, window_stride=3, **args):

        self.images_path = []
        self.poses = []
        self.intrinsics = []
        self.width = 1920
        self.height = 1280
        self.data_dir = dataset_path
        self.min_depth = 0.1
        self.data_index = []

        sensor_type = ['cam_1', 'cam_2', 'cam_3']

        poses_per_camera = [[] for step in range(5)]
        intrinsics_per_camera = [[] for step in range(5)]
        cam_idx_dict = {'camera_FRONT':0, 'camera_FRONT_LEFT':1, 'camera_FRONT_RIGHT':2, 'camera_SIDE_LEFT':3, 'camera_SIDE_RIGHT': 4}

        images_root = os.path.join(self.data_dir, 'images')
        scene_info_path = os.path.join(self.data_dir, 'scenario.pt')
        with open(scene_info_path, 'rb') as f:
            scenario = pickle.load(f)
        for oid, odict in scenario['observers'].items():
            if (o_class_name:=odict['class_name']) == 'Camera':
                intr = odict['data']['intr']
                c2w = odict['data']['c2w']
                intrinsics_per_camera[cam_idx_dict[oid]].append(intr)
                poses_per_camera[cam_idx_dict[oid]].append(c2w)

        poses_json_path = os.path.join(self.data_dir,'scene_after_pr/mvs_driving_all_pose/sparse/0/pose.json')
        with open(poses_json_path) as jp:
            poses_json = json.load(jp)

        video_lens = 80
        for idx in range(video_lens):
            for cam_idx, cam in enumerate(sensor_type):
                rgb_path = os.path.join(images_root, cam, str(idx).zfill(8)+'.jpg')
                # pose_cam2world = poses_per_camera[cam_idx][0][idx]
                # pose_world2cam = np.linalg.inv(pose_cam2world)
                intrinsic = intrinsics_per_camera[cam_idx][0][idx]

                self.images_path.append(rgb_path)
                # self.poses.append(pose_world2cam)
                self.intrinsics.append(intrinsic)
                self.data_index.append(str(idx).zfill(8)+cam)

                pose_key_2 = rgb_path.split("/")[-1][:-4]
                pose_key_1 = rgb_path.split("/")[-2]
                pose_key = pose_key_1 + "/" + pose_key_2
                pose_attrs = poses_json[pose_key]
                quat = [pose_attrs['q_x'], pose_attrs['q_y'], pose_attrs['q_z'], pose_attrs['q_w']]
                pose_world2cam = np.eye(4)
                Rot = R.from_quat(np.array(quat)).as_matrix()
                pose_world2cam[:3, 3] = np.array([pose_attrs['p_x'], pose_attrs['p_y'], pose_attrs['p_z']])
                pose_world2cam[:3, :3] = Rot
                self.poses.append(pose_world2cam)

        self.num_frames = num_frames
        self.image_format = rgb_path[0][-3:]
        self.offsets = np.array([-3*len(sensor_type), -2*len(sensor_type), -1*len(sensor_type), len(sensor_type), 2*len(sensor_type), 3*len(sensor_type)])
        self.window_stride = window_stride


    def __len__(self):
        return len(self.poses)
    
    def __getitem__(self, index):
        indices = self.offsets.copy() + index
        while indices[0] < 0:
            indices += self.window_stride
        while indices[-1] >= len(self.poses):
            indices -= self.window_stride
        assert(indices[0] >= 0)
        indices = [index] + [i for i in indices if i != index]
        images, poses, intrinsics = [], [], []
        for i in indices:
            image = read_gen(self.images_path[i])
            images.append(image)
            poses.append(self.poses[i])
            intrinsics.append(self.intrinsics[i])

        #scale = 400 / self.min_depth
        scale = 200

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

