from pickle import FALSE, TRUE
import sys
import os
from xml.etree.ElementTree import PI
import cv2
import json
import logging
import copy
import numpy as np
from tqdm import tqdm
from enum import Enum
import functools
import pickle
import matplotlib.pyplot as plt
import matplotlib
from nuscenes.nuscenes import NuScenes
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
import shutil

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

pipeline_script_dir = sys.path[0]
root_dir = pipeline_script_dir.split("scripts/")[0]
logging.info(f"pipeline_script_dir={pipeline_script_dir}")
logging.info(f"root_dir={root_dir}")
sys.path.append(root_dir+"/scripts/")

import argparse
import time
import math
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from shutil import rmtree
import shutil
from scipy.spatial.transform import Rotation as R
from python.read_write_model import write_images_binary, write_images_text, write_cameras_binary, write_cameras_text, write_points3D_text
from database import COLMAPDatabase, camModelDict
from SuperPointDetectors import get_super_points_from_scenes_return
from matchers import mutual_nn_ratio_matcher, mutual_nn_matcher
import yaml

class ViewType(Enum):
    ThreeView = 0
    CENTERVIEW = 1
    LEFTVIEW = 2
    RIGHTVIEW = 3

def get_image_name_list(db: COLMAPDatabase, camera_id: int=0):
    name_list = []
    entries = db.execute("SELECT * FROM images")
    for img_id, name, cam_id, qe, qx, qy, qz, tx, ty, tz in entries:
        if cam_id == camera_id:
            name_list.append(name)
        
    return name_list

def get_camera_id_list(db: COLMAPDatabase):
    id_list = []
    entries = db.execute("SELECT * FROM cameras")
    for camera_id, model, width, height, params, prior_f in entries:
        id_list.append(camera_id)
    return id_list

def import_feature(db, database_root, feature_type="SuperPoint"):
    print("feature extraction .......................")

    image_name_list = list()

    for cam_id in get_camera_id_list(db):
        image_name_list += get_image_name_list(db, cam_id)

    images_path = [os.path.join(database_root, x) for x in image_name_list]

    if feature_type == "SuperPoint":
        res = get_super_points_from_scenes_return(images_path)
    elif feature_type == "SIFT":
        print("error: no implementation in SIFT")

    db.execute("DELETE FROM keypoints;")
    db.execute("DELETE FROM descriptors;")
    db.execute("DELETE FROM matches;")
    db.execute("DELETE FROM two_view_geometries;")
    for image_id, kp in enumerate(res):
        keypoints = kp["keypoints"]
        scores = kp["scores"].flatten()
        keypoints = keypoints[:, :2]
        keypoints = np.concatenate(
            (keypoints, scores.reshape(-1, 1), scores.reshape(-1, 1)), axis=1)
        keypoints = keypoints.astype(np.float32)
        db.add_keypoints(image_id, keypoints)
    
    return res


def make_matching_figure(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplot(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap="gray")
    axes[1].imshow(img1, cmap="gray")
    for i in range(2):
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    #draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                             (fkpts0[i, 1], fkpts1[i, 1]),
                                             transform=fig.transFigure, c=color[i], linewidth=1)]
        
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color
    )

    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()

def match_features_bruce(db, keypoints, database_root, use_superglue=True, match_overlap=5, image_paths=None):
    print("match features by bruce force matching....")

    RANSAC_THRESHOLD = 5
    MIN_NUM_INLIERS = 30
    MIN_INLIER_RATIO = 0.3

    camera_id = 0
    image_name_list = get_image_name_list(db, camera_id=camera_id)

    # squential match
    num_images = len(image_name_list)

    assert len(keypoints) % num_images == 0

    image_path = os.path.join(database_root, image_name_list[0])
    image_height, image_width = cv2.imread(image_path).shape[:2]

    step_range = [*range(1, match_overlap)]

    total_iters = 0
    for step in step_range:
        for i in range(0, num_images - step):
            total_iters += 1
    
    # if use_superglue:
    #     superglue_matcher = SuperGlueMatcher()

    cam_id_list = get_camera_id_list(db)
    cam_name = ['cam_1', 'cam_2', 'cam_3']
    iter_num = len(cam_id_list)
    pbar = tqdm(total=total_iters)
    for idx_i in range(num_images*3):
        for idx_j in range(idx_i+1, num_images*3):
            if use_superglue:
                data = {"descriptors0": keypoints[idx_i]['descriptors'],
                        "descriptors1": keypoints[idx_j]['descriptors'],
                        "keypoints0": keypoints[idx_i]['keypoints'],
                        "keypoints1": keypoints[idx_j]['keypoints'],
                        "image0": np.zeros((image_height, image_width), dtype=np.uint8),
                        "image1": np.zeros((image_height, image_width), dtype=np.uint8),
                        "scores0": keypoints[idx_i]["scores"],
                        "scores1": keypoints[idx_j]["scores"]}
                
                # matches = superglue_matcher(data)
                RANSAC_THRESHOLD = 8
            else:
                D1 = keypoints[idx_i]["descriptors"] * 1.0
                D2 = keypoints[idx_j]["descriptors"] * 1.0
                matches = mutual_nn_ratio_matcher(D1, D2, ratio=0.95).astype(np.uint32)

            if matches.shape[0] > 0:
                db.add_matches(idx_i, idx_j, matches)

            if matches.shape[0] > MIN_NUM_INLIERS:
                # two view geometry
                kp1 = keypoints[idx_i]['keypoints']
                kp2 = keypoints[idx_j]['keypoints']
                F, mask_f = cv2.findFundamentalMat(kp1[matches[:, 0], :2],
                                                   kp2[matches[:, 1], :2],
                                                   cv2.FM_RANSAC,
                                                   RANSAC_THRESHOLD, 0.99, 2000)
                H, mask_h = cv2.findHomography(kp1[matches[:, 0], :2],
                                               kp2[matches[:, 1], :2],
                                               cv2.FM_RANSAC,
                                               RANSAC_THRESHOLD, 0.99, 2000)
                
                if ((np.sum(mask_f) >= MIN_NUM_INLIERS and np.mean(mask_f) >= MIN_INLIER_RATIO) 
                    or (np.sum(mask_h) >= MIN_NUM_INLIERS and np.mean(mask_h) >= MIN_INLIER_RATIO)):
                    if np.sum(mask_f) < np.sum(mask_h):
                        mask = mask_h
                        mask = mask.astype(np.bool).flatten()
                        db.add_two_view_geometry(idx_i, idx_j, matches[mask, :],
                                                 F=np.eye(3), E=np.eye(3), H=H, config=6)
                    else:
                        mask = mask_f
                        mask = mask.astype(np.bool).flatten()
                        db.add_two_view_geometry(idx_i, idx_j, matches[mask, :],
                                                 F, E=np.eye(3), H=np.eye(3), config=3)
            pbar.update(1)
    pbar.close()

def match_features(db, keypoints, database_root, use_superglue=True, match_overlap=5, image_paths=None):
    print("match features by sequential matching....")

    RANSAC_THRESHOLD = 5
    MIN_NUM_INLIERS = 30
    MIN_INLIER_RATIO = 0.3

    camera_id = 0
    image_name_list = get_image_name_list(db, camera_id=camera_id)

    # squential match
    num_images = len(image_name_list)

    assert len(keypoints) % num_images == 0

    image_path = os.path.join(database_root, get_image_name_list[0])
    image_height, image_width = cv2.imread(image_path).shape[:2]

    step_range = [*range(1, match_overlap)]

    total_iters = 0
    for step in step_range:
        for i in range(0, num_images - step):
            total_iters += 1
    
    # if use_superglue:
    #     superglue_matcher = SuperGlueMatcher()

    cam_id_list = get_camera_id_list(db)
    cam_name = ['cam_1', 'cam_2', 'cam_3']
    iter_num = len(cam_id_list)
    pbar = tqdm(total=total_iters)
    for cam_id, in range(iter_num):
        for step in step_range:
            for i in range(0, num_images-step):
                if use_superglue:
                    data = {"descriptors0": keypoints[num_images * cam_id + i]['descriptors'],
                            "descriptors1": keypoints[num_images * cam_id + i + step]['descriptors'],
                            "keypoints0": keypoints[num_images * cam_id + i]['keypoints'],
                            "keypoints1": keypoints[num_images * cam_id + i + step]['keypoints'],
                            "image0": np.zeros((image_height, image_width), dtype=np.uint8),
                            "image1": np.zeros((image_height, image_width), dtype=np.uint8),
                            "scores0": keypoints[num_images * cam_id + i]["scores"],
                            "scores1": keypoints[num_images * cam_id + i + step]["scores"]}
                    
                    # matches = superglue_matcher(data)
                    RANSAC_THRESHOLD = 8
                else:
                    D1 = keypoints[num_images * cam_id + i]["descriptors"] * 1.0
                    D2 = keypoints[num_images * cam_id + i + step]["descriptors"] * 1.0
                    matches = mutual_nn_ratio_matcher(D1, D2, ratio=0.95).astype(np.uint32)

                if matches.shape[0] > 0:
                    db.add_matches(num_images * cam_id + i, num_images * cam_id + i + step, matches)

                if matches.shape[0] > MIN_NUM_INLIERS:
                    # two view geometry
                    kp1 = keypoints[num_images * cam_id + i]['keypoints']
                    kp2 = keypoints[num_images * cam_id + i + step]['keypoints']
                    F, mask_f = cv2.findFundamentalMat(kp1[matches[:, 0], :2],
                                                    kp2[matches[:, 1], :2],
                                                    cv2.FM_RANSAC,
                                                    RANSAC_THRESHOLD, 0.99, 2000)
                    H, mask_h = cv2.findHomography(kp1[matches[:, 0], :2],
                                                kp2[matches[:, 1], :2],
                                                cv2.FM_RANSAC,
                                                RANSAC_THRESHOLD, 0.99, 2000)
                    
                    if ((np.sum(mask_f) >= MIN_NUM_INLIERS and np.mean(mask_f) >= MIN_INLIER_RATIO) 
                        or (np.sum(mask_h) >= MIN_NUM_INLIERS and np.mean(mask_h) >= MIN_INLIER_RATIO)):
                        if np.sum(mask_f) < np.sum(mask_h):
                            mask = mask_h
                            mask = mask.astype(np.bool).flatten()
                            db.add_two_view_geometry(num_images * cam_id + i, num_images * cam_id + i + step, matches[mask, :],
                                                    F=np.eye(3), E=np.eye(3), H=H, config=6)
                        else:
                            mask = mask_f
                            mask = mask.astype(np.bool).flatten()
                            db.add_two_view_geometry(num_images * cam_id + i, num_images * cam_id + i + step, matches[mask, :],
                                                    F, E=np.eye(3), H=np.eye(3), config=3)
                pbar.update(1)
    pbar.close()

def get_init_cameraparams(width, height, modelId=0, camera_params=None):
    if camera_params is None:
        f = max(width, height) * 1.2
        cx = width / 2.0
        cy = height / 2.0
    else:
        f = (camera_params["fx"] + camera_params["fy"]) / 2.0
        cx = camera_params["cx"]
        cy = camera_params["cy"]
                           
    if modelId == 0:
        return np.array([f, cx, cy])
    elif modelId == 1:
        return np.array([f, f, cx, cy])
    elif modelId == 2 or modelId == 8:
        return np.array([f, cx, cy, 0.0])
    elif modelId == 3 or modelId == 7:
        return np.array([f, cx, cy, 0.0, 0.0])
    elif modelId == 4 or modelId == 5:
        return np.array([(f, f, cx, cy, 0.0)])
    elif modelId == 9:
        return np.array([f, f, cx, cy, 0.0])
    return np.array([f, f, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

def convert_pose_calib_for_colmap(db, all_odo, calib_list, model_path, pinhole_cams_list, image_names):
    count = 0
    print('all select odos: ', len(all_odo['cam_1']))
    import copy
    select_pose = copy.deepcopy(all_odo)
    odo_base = np.linalg.inv(select_pose['cam_1'][0])
    for idx in range(len(select_pose['cam_1'])):
        for cam in pinhole_cams_list:
            select_pose[cam][idx] = np.linalg.inv(odo_base @ select_pose[cam][idx])
        
    # save images
    frame_num = len(select_pose['cam_1'])
    images_file = os.path.join(model_path, "images.txt")
    HEADER = '# Images list with two lines of data per image:n' + \
            "#   Images_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n" + \
            "#   POINTS2D[] AS (X, Y, POINT3D_ID)\n" + \
            "# Number of images: {}, mean obersevation per image: {}\n".format(len(select_pose['cam_1']) * 3, 0)
    with open(images_file, "w") as fid:
        fid.write(HEADER)
        for img_id in range(len(select_pose['cam_1'])):
            count = 0
            for cam_id in pinhole_cams_list:
                world_to_cam = select_pose[cam_id][img_id]

                ###########
                #Waymo#
                ###########
                image_name = image_names[cam_id][img_id].split('/')[-2] + '/' + image_names[cam_id][img_id].split('/')[-1]
                ##########
                #Nuscenes#
                ##########
                #image_name = cam_id + '/' + image_names[cam_id][img_id].split('/)[-1]
                ##############
                r = R.from_matrix(world_to_cam[:3, :3])
                qvec = r.as_quat()
                tvec = world_to_cam[:3, 3]
                image_header = [img_id + frame_num * count, qvec[3], qvec[0], qvec[1], qvec[2], 
                                tvec[0], tvec[1], tvec[2], count, image_name]
                first_line = " ".join(map(str, image_header))
                fid.write(first_line + "\n")
                fid.write("\n")

                db.add_image(image_name, count, image_id=int(img_id + frame_num * count))
                count += 1

    #save cameras
    cameras_file = os.path.join(model_path, "cameras.txt")

    HEADER = "# Camera list with open line of data per camera:\n" + \
            "#  CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[] \n" + \
            "# Number of cameras: {}\n".format(1)
    
    with open(cameras_file, "w") as fid:
        fid.write(HEADER)
        count = 0
        for cam_id in pinhole_cams_list:
            fx = calib_list[cam_id]["fx"]
            fy = calib_list[cam_id]["fy"]
            cx = calib_list[cam_id]["cx"]
            cy = calib_list[cam_id]["cy"]
            if cam_id in ['cam_1', 'cam_2', 'cam_3']:
                width, height = 1920, 1280
                #width, height = 1600, 900
            else:
                width, height = 1920, 1280
            to_write = [count, "SIMPLE_PINHOLE", width, height, fx, cx, cy]
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")

            #save cacmera db
            camera_params = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
            cameraModel = camModelDict['SIMPLE_PINHOLE']
            params = get_init_cameraparams(width, height, cameraModel, camera_params)
            print("Camera type = ", cameraModel)
            print("Camera parameters = ", params)
            db.add_camera(cameraModel, width, height, params, camera_id=count, prior_focal_length=True)
            count += 1

    write_points3D_text({}, os.path.join(model_path, "points3D.txt"))

def generate_cam_rigid_config(db, database_root, pinhole_cams_list, calib_list):
    cam_rigid = dict()
    cam_id_list = get_camera_id_list(db)

    cam_rigid["ref_camera_id"] = cam_id_list[0]
    rigid_cam_list = []

    count = 0
    for cam_id in pinhole_cams_list:
        rigid_cam = dict()
        rigid_cam["camera_id"] = count
        rigid_rot = np.eye(4)
        center_cam_to_ego = np.asmatrix(calib_list['cam_1'][0])
        cam_to_ego = np.asmatrix(calib_list[cam_id][0])
        rigid_rot = np.linalg.inv(cam_to_ego) @ center_cam_to_ego
        print(rigid_rot)
        r = R.from_matrix(rigid_rot[:3, :3])
        qvec = r.as_quat()
        rigid_cam["image_prefix"] = str(cam_id)
        rigid_cam["rel_tvec"] = [rigid_rot[0, 3]]
        rigid_cam["rel_qvec"] = [qvec[3], qvec[0], qvec[1], qvec[2]]
        rigid_cam_list.append(rigid_cam)
        count += 1

    cam_rigid["cameras"] = rigid_cam_list

    out_json_path = os.path.join(database_root, "cam_rigid_config.json")
    with open(out_json_path, "w+") as f:
        json.dump([cam_rigid], f, indent=4)

def load_vins(odo_path):
    odo_list = os.listdir(odo_path)
    odo_list.sort(key = lambda x: int(x[:-4]))
    all_odo = {}
    for odo_file in odo_list:
        time_stamp = odo_file.split('.')[0]
        config_path = os.path.join(odo_path, odo_file)
        ego_to_world = np.loadtxt(config_path)
        np.asmatrix(ego_to_world)
        all_odo[time_stamp] = ego_to_world

    return all_odo

def load_odometry_json(vins_json_path, all_time_stamp):
    all_odo = {}
    with open(vins_json_path, "r") as file:
        vins_json = json.load(file)
        for time_stamp in vins_json:
            if not time_stamp in all_time_stamp:
                continue
        tmp = vins_json[time_stamp]['imu_to_world']
        ego_to_world = np.eye(4)
        count = 0
        for i in range(3):
            for j in range(4):
                ego_to_world[i, j] = tmp[count]
                count = count + 1
            all_odo[time_stamp] = ego_to_world

    MIN_DIST_THR = 0.05
    last_posi = None
    select_odos = {}
    for time_stamp in all_odo:
        ego_to_world = all_odo[time_stamp]
        if last_posi is None:
            last_posi = np.linalg.inv(ego_to_world)
            select_odos[time_stamp] = ego_to_world
        else:
            cur_to_last = last_posi @ ego_to_world
            if np.linalg.norm(cur_to_last[:3, 3]) > MIN_DIST_THR:
                last_posi = np.linalg.inv(ego_to_world)
                select_odos[time_stamp] = ego_to_world

    return select_odos


def load_odometry(odo_path, all_time_stamp):
    odo_list = os.listdir(odo_path)
    odo_list.sort()

    all_odo = {}
    for odo_file in odo_list:
        time_stamp = odo_file.split('.')[0]
        if not time_stamp in all_time_stamp:
            continue
    
        config_path = os.path.join(odo_path, odo_file)
        fhand = open(config_path, "rt")
        ego_to_world = np.eye(4)
        for line in fhand:
            tmp = line.split(' ')
            count = 0
            for i in range(3):
                for j in range(4):
                    ego_to_world[i, j] = tmp[count+1]
                    count = count + 1
            all_odo[time_stamp] = ego_to_world
    MIN_DIST_THR = 0.05
    last_posi = None
    select_odos = {}
    for time_stamp in all_odo:
        ego_to_world = all_odo[time_stamp]
        if last_posi is None:
            last_posi = np.linalg.inv(ego_to_world)
            select_odos[time_stamp] = ego_to_world
        else:
            cur_to_last = last_posi @ ego_to_world
            if np.linalg.norm(cur_to_last[:3, 3]) > MIN_DIST_THR:
                last_posi = np.linalg.inv(ego_to_world)
                select_odos[time_stamp] = ego_to_world
    
    return select_odos


def load_calib(calib_path, intri_name, calib_name):
    fhand = open(calib_path, "rt")
    calib = {}
    for line in fhand:
        if (intri_name in line) and ('crop' not in line) and ('resize' not in line):
            tmp = line.split(' ')
            calib['fx'] = float(tmp[1])
            calib['fy'] = float(tmp[6])
            calib['cx'] = float(tmp[3])
            calib['cy'] = float(tmp[7])
        if calib_name in line:
            tmp = line.split(' ')
            cam_to_ego = np.eye(4)
            count = 0
            for i in range(3):
                for j in range(4):
                    cam_to_ego[i, j] = tmp[count + 1]
                    count = count + 1
            calib['cam_to_ego'] = cam_to_ego
    return calib


def load_opt_calib(calib_path, cam_id):
    calib = {}
    with open(calib_path, "r") as file:
        calib_json = json.load(file)
        for tiem in calib_json:
            tmp = calib_json[time]['cam' + cam_id + '_to_ego']
            cam_to_ego = np.eye(4)
            count = 0
            for i in range(3):
                for j in range(4):
                    cam_to_ego[i, j] = tmp[0][count]
                    count = count + 1
            calib['cam_to_ego'] = cam_to_ego
            break
    return calib

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

    parser = argparse.ArgumentParser(description='feature extraction and matching!')

    parser.add_argument('--origin_path', required=True, type=str, help='origin_path.')
    parser.add_argument('--input_path', required=True, type=str, help='input_path.')
    parser.add_argument('---db_path', required=False, type=str, default=None, help='db_path.')

    args = parser.parse_args()
    if args.db_path is None:
        args.db_path = args.input_path

    images_path = {'cam_1': [], 'cam_2': [], 'cam_3': []}
    all_odo = {'cam_1': [], 'cam_2': [], 'cam_3': []}
    calib_list = {'cam_1': {}, 'cam_2': {}, 'cam_3': {}}
    ##################
    #######Waymo######
    ##################
    width = 1920
    height = 1280
    pinhole_cams_list = ['cam_1', 'cam_2', 'cam_3']
   
    poses_per_camera = [[] for step in range(5)]
    intrinsics_per_camera = [[] for step in range(5)]
    cam_idx_dict = {'camera_FRONT':0, 'camera_FRONT_LEFT':1, 'camera_FRONT_RIGHT':2, 'camera_SIDE_LEFT':3, 'camera_SIDE_RIGHT': 4}

    images_root = os.path.join(args.origin_path, 'images')
    imgs_path = os.listdir(images_root)
    for ip in imgs_path:
        shutil.copytree(os.path.join(images_root, ip), os.path.join(args.input_path, ip))
    
    sky_segments_root = os.path.join(args.origin_path, 'masks')
    depth_root = os.path.join(args.origin_path, 'depths')
    scene_info_path = os.path.join(args.origin_path, 'scenario.pt')
    with open(scene_info_path, 'rb') as f:
        scenario = pickle.load(f)
    for oid, odict in scenario['observers'].items():
        if (o_class_name:=odict['class_name']) == 'Camera':
            intr = odict['data']['intr']
            c2w = odict['data']['c2w']
            intrinsics_per_camera[cam_idx_dict[oid]].append(intr)
            poses_per_camera[cam_idx_dict[oid]].append(c2w)

    video_lens = 80
    sensor_type = ['cam_1', 'cam_2', 'cam_3']
    for idx in range(video_lens):
        for cam_idx, cam in enumerate(sensor_type):
            rgb_path = os.path.join(images_root, cam, str(idx).zfill(8)+'.jpg')
            depth_path = os.path.join(depth_root, cam, str(idx).zfill(8)+'.npz')
            segment_path = os.path.join(sky_segments_root, cam, str(idx).zfill(8)+'.npz')
            pose_cam2world = poses_per_camera[cam_idx][0][idx]
            intrinsic = intrinsics_per_camera[cam_idx][0][idx]

            calib_list[cam]['fx'] = intrinsic[0][0]
            calib_list[cam]['fy'] = intrinsic[1][1]
            calib_list[cam]['cx'] = intrinsic[0][2]
            calib_list[cam]['cy'] = intrinsic[1][2]

            all_odo[cam].append(pose_cam2world)
            images_path[cam].append(rgb_path)

   
    # temp_dir = os.path.join(args.origin_path, 'cam_1')
    # pickle_idxes = sorted(os.listdir(temp_dir))
    # prefix_path = ''
    # scene = args.origin_path.split("/")[-2]
    # for pickle_idx in pickle_idxes:
    #     for cam in pinhole_cams_list:
    #         pickle_path = os.path.join(args.origin_path, cam, pickle_idx)
    #         with open(pickle_path, 'rb') as f:
    #             data = pickle.load(f)
    #             rgb_path = os.path.join(prefix_path, data['rgb'])
    #             focal = np.array(data['cam_in'])
    #             pose = np.array(data['pose']).reshape(4, 4)
    #             cam_ex = np.array(data['cam_ex']).reshape(4, 4)

    #             calib_list[cam]['fx'] = focal[0]
    #             calib_list[cam]['fy'] = focal[1]
    #             calib_list[cam]['cx'] = focal[2]
    #             calib_list[cam]['cy'] = focal[3]

    #             pose_cam2world = pose @ cam_ex
    #             all_odo[cam].append(pose_cam2world)

    #             images_path[cam].append(rgb_path)

    ################################################
    ################################################

    ###############################################
    #########Nuscenes##########
    ###############################################
    # width = 1600
    # height = 900
    # nusc = NuScenes(version='v1.0-trainval', dataroot='/opt/ml/project')
    # scene_name = args.origin_path
    # for scene in nusc.scene:
    #     if scene['name'] == scene_name:
    #         break
    # pinhole_cams_list_ori = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
    # pinhole_cams_list = ['cam_1', 'cam_2', 'cam_3']
    # IDX = 0
    # sample_idx_list = {}
    # scene_token = scene['token']
    # temp_sample = nusc.get('sample', scene['first_sample_token'])
    # for s_idx, s in enumerate(pinhole_cams_list_ori):
    #     temp_data = nusc.get('sample_data', temp_sample['data'][s])
    #     cam = pinhole_cams_list[s_idx]
    #     for i in range(120):
    #         rgb_path, _, cam_intrinsic = nusc.get_sample_data(temp_data['token'])
    #         if (temp_data['is_key_frame']):
    #             sample_idx_list[IDX] = temp_data['token']
    #         IDX += 1

    #         images_path[cam].append(rgb_path)

    #         if not os.path.exists(rgb_path):
    #             temp_data = nusc.get('sample_data', temp_data['next'])
    #             print("no data")
    #             continue

    #         #intrinsic
    #         calib_list[cam]['fx'] = cam_intrinsic[0, 0]
    #         calib_list[cam]['fy'] = cam_intrinsic[1, 1]
    #         calib_list[cam]['cx'] = cam_intrinsic[0, 2]
    #         calib_list[cam]['cy'] = cam_intrinsic[1, 2]           

            # cam2ego_r = Quaternion(temp_cam2ego['rotation']).rotation_matrix
            # cam2ego_t = np.array(temp_cam2ego['translation'])
            # cam2ego_rt = np.eye(4)
            # cam2ego_rt[:3, :3] = cam2ego_r
            # cam2ego_rt[:3, 3] = cam2ego_t
            # pose_cam2world = ego2global_rt @ cam2ego_rt
            # all_odo[cam].append(pose_cam2world)

            #temp_data = nusc.get('sample_data', temp_data['next'])

    ########################################################
    ########################################################

    target_path = 'mvs_driving_all_pose'
    model_path = os.path.join(args.input_path, target_path, 'sparse', '0')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    database_path = os.path.join(args.db_path, target_path, 'database.db')
    print("database_path = %s" % (database_path))
    if not os.path.exists(os.path.join(args.db_path, target_path)):
        os.makedirs(os.path.join(args.db_path, target_path))

    if os.path.exists(database_path):
        print("database exists")
        os.remove(database_path)

    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    convert_pose_calib_for_colmap(db, all_odo, calib_list, model_path, pinhole_cams_list, images_path)

    generate_cam_rigid_config(db, os.path.join(args.input_path, target_path), pinhole_cams_list, all_odo)

    feature_type = "SuperPoint"
    if os.path.exists(os.path.join(args.origin_path, 'frames', 'rect_cam_0')):
        res = import_feature(db, os.path.join(args.origin_path, 'frames'), feature_type)
    else:
        res = import_feature(db, args.input_path, feature_type)

    match_overlap = 8
    match_features_bruce(db, res, args.input_path, use_superglue=0, match_overlap=match_overlap, image_paths=images_path)

    db.commit()
    db.close()