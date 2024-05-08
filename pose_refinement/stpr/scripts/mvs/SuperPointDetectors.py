from typing import Any
from superpoint import SuperPoint
import cv2
import numpy as np
import torch
import json
import argparse
from tqdm import tqdm
import os

def image2tensor(frame, device):
    return torch.from_numpy(frame / 255.).float()[None, None].to(device)

class SuperPointDetector(object):
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
        "path": "superpoint_v1.pth",
        "cuda": True
    }

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        print("SuperPoint detector config: ")
        print(self.config)

        self.device = 'cuda' if torch.cuda.is_available() and self.config["cuda"] else 'cpu'

        print("creating SuperPoint detector...")
        self.superpoint = SuperPoint(self.config).to(self.device)
        self.superpoint.eval()

    def __call__(self, image):
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_tensor = image2tensor(image, self.device)
        with torch.no_grad():
            pred = self.superpoint({'image': image_tensor})

        ret_dict = {
            "image_size": [image.shape[0], image.shape[1]],
            #"torch": pred,
            "keypoints": pred["keypoints"][0].cpu().detach().numpy(),
            "scores": pred["scores"][0].cpu().detach().numpy(),
            "descriptors": pred["descriptors"][0].cpu().detach().numpy().transpose()
        }
        
        return ret_dict
        
def get_super_points_from_scenes_return(images_path):
    spd = SuperPointDetector()
    out_list = []
    for path in tqdm(images_path):
        image = cv2.imread(path)
        if image is None:
            print("%s dose not exist:"%(path))
        ret_dict = spd(image)
        out_list.append(ret_dict)
    return out_list
