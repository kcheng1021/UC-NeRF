from pickle import FALSE, TRUE
import sys
import os
from xml.etree.ElementTree import PI
import logging
import copy
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R

pipeline_script_dir = sys.path[0]
root_dir = pipeline_script_dir.split("scripts/")[0]
logging.info(f"pipeline_script_dir={pipeline_script_dir}")
logging.info(f"root_dir={root_dir}")
sys.path.append(root_dir+"/scripts/")

from python.read_write_model import read_cameras_binary, read_cameras_text, read_images_binary, write_images_binary, read_points3D_binary, read_points3D_text, write_points3D_binary, read_images_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="pose_scale_correct!")

    parser.add_argument('--sparse_path', required=True, type=str, help='sparse_path.')

    args = parser.parse_args()

    if os.path.exists(os.path.join(args.sparse_path, 'images.bin')):
        opt_images = read_images_binary(os.path.join(args.sparse_path, "images.bin"))
        cameras = read_cameras_binary(os.path.join(args.sparse_path, "cameras.bin"))
        opt_points_3d = read_points3D_binary(os.path.join(args.sparse_path, "points3D.bin"))
    else:
        opt_images = read_images_text(os.path.join(args.sparse_path, "images.txt"))
        cameras = read_cameras_text(os.path.join(args.sparse_path, "cameras.txt"))
        opt_points_3d = read_points3D_text(os.path.join(args.sparse_path, "points3D.txt"))
        write_images_binary(opt_images, os.path.join(args.sparse_path, "images.bin"))
        write_points3D_binary(opt_points_3d, os.path.join(args.sparse_path, "points3D.bin"))
        exit(0)
    vins_images = read_images_text(os.path.join(args.sparse_path, "images.txt"))

    img_num = int(len(opt_images) / len(cameras))
    opt_tvecs = []
    vins_tvecs = []
    for idx in range(img_num):
        opt_odo = np.eye(4)
        quat = opt_images[idx].qvec
        opt_odo[:3, :3] = R.from_quat([*quat[1:], quat[0]]).as_matrix()
        opt_odo[:3, 3] = opt_images[idx].tvec
        opt_tvecs.append(np.linalg.inv(opt_odo)[:3, 3])
        vins_odo = np.eye(4)
        quat = vins_images[idx].qvec
        vins_odo[:3, :3] = R.from_quat([*quat[1:], quat[0]]).as_matrix()
        vins_odo[:3, 3] = vins_images[idx].tvec
        vins_tvecs.append(np.linalg.inv(vins_odo)[:3, 3])

    opt_dists = 0.0
    vins_dist = 0.0
    for i in range(img_num - 1):
        dist1 = np.linalg.norm(opt_tvecs[i] - opt_tvecs[i+1])
        opt_dists += dist1
        dist2 = np.linalg.norm(vins_tvecs[i] - vins_tvecs[i+1])
        vins_dist += dist2
    mean_scales = vins_dist / opt_dists
    print("scales: ", mean_scales, vins_dist, opt_dists)

    for id, img in opt_images.items():
        tmp = img.tvec * mean_scales
        opt_images[id] = img._replace(tvec=tmp)

    for id, point in opt_points_3d.items():
        tmp = point.xyz * mean_scales
        opt_points_3d[id] = point._replace(xyz=tmp)

    write_images_binary(opt_images, os.path.join(args.sparse_path, "images.bin"))
    write_points3D_binary(opt_points_3d, os.path.join(args.sparse_path, "points3D.bin"))
            