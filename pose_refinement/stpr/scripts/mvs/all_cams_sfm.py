from pickle import FALSE, TRUE
import sys
import os
from xml.etree.ElementTree import PI
import logging
import copy
import numpy as np
import argparse
import time
import json
import functools
from distutils.dir_util import copy_tree
from scipy.spatial.transform import Rotation as R

pipeline_script_dir = sys.path[0]
root_dir = pipeline_script_dir.split("scripts/")[0]
logging.info(f"pipeline_script_dir={pipeline_script_dir}")
logging.info(f"root_dir={root_dir}")
sys.path.append(root_dir+"/scripts")

from python.read_write_model import read_images_binary, read_cameras_binary, read_points3D_binary, write_camera_trajectory_json

def load_pose_josn(pose_path, cam_id):
    all_pose = []
    with open(pose_path, "r") as file:
        pose_json = json.load(file)
        for time_stamp in pose_json:
            if not "image_" + cam_id in time_stamp:
                continue
            q_w = pose_json[time_stamp]['q_w']
            q_x = pose_json[time_stamp]['q_x']
            q_y = pose_json[time_stamp]['q_y']
            q_z = pose_json[time_stamp]['q_z']
            p_x = pose_json[time_stamp]['p_x']
            p_y = pose_json[time_stamp]['p_y']
            p_z = pose_json[time_stamp]['p_z']
            world_to_cam = np.eye(4)
            world_to_cam[:3, :3] = R.from_quat([q_x, q_y, q_z, q_w]).as_matrix()
            world_to_cam[:3, 3] = [p_x, p_y, p_z]
            all_pose.append(world_to_cam)
    return all_pose

def my_comp(a, b):
    x = int(a)
    y = int(b)
    if x < y:
        return -1
    elif y < x:
        return 1
    else:
        return 0
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='pinhole mvs pipline!')

    parser.add_argument('--origin_path', required=True, type=str, help='origin_path.')
    parser.add_argument("--input_path", required=True, type=str, help="input_path.")
    parser.add_argument("--use_opt_calib", required=False, type=int, default=0, help='use_opt_calib')
    parser.add_argument("--fix_trans_refine_rot", required=False, type=int, default=0, help='fix_trans_refine_rot')

    args = parser.parse_args()
    if not os.path.exists(args.input_path):
        os.makedirs(args.input_path)

    time_log_path = os.path.join(args.input_path, 'time_log_sfm.txt')
    time_log = open(time_log_path, "w")

    #feature matching
    start_time = time.time()
    cmd = f"python prepare_all_data_for_mvs.py --origin_path {args.origin_path} --input_path {args.input_path}"
    os.system(cmd)
    end_time = time.time()
    feature_extraction_and_matching_time = end_time - start_time
    time_log.write("feature_extraction_and_matching_time: " + str(feature_extraction_and_matching_time) + "s\n")

    #run sfm
    start_time = time.time()
    colmap_exe_dir = root_dir + "build/src/exe/colmap"
    target_path = "mvs_driving_all_pose"
    target_path = os.path.join(args.input_path, target_path)
    pose_scale_path = root_dir + "/scripts/mvs/"
    cmd = f"bash run_multi_cam_pose_opt.sh {colmap_exe_dir} {target_path} {target_path} {args.input_path} {pose_scale_path} {args.fix_trans_refine_rot}"
    os.system(cmd)

    end_time = time.time()
    sfm_and_dense_time = end_time - start_time
    time_log.write("sfm_and_dense_time: " + str(sfm_and_dense_time) + "s\n")

    ext = '.bin'
    images = read_images_binary(os.path.join(target_path, "sparse", "0", "images" + ext))
    write_camera_trajectory_json(os.path.join(target_path, "sparse", "0"), images)

    time_log.close()