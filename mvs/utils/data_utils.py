import numpy as np
import gin
import torch.nn.functional as F


@gin.configurable()
def random_scale_and_crop(images, depths, intrinsics, crop_size=[1056, 1440], smin=-0.15, smax=0.5):
    s = 2 ** np.random.uniform(smin, smax)

    ht1 = images.shape[2]
    wd1 = images.shape[3]
    ht2 = int(s * ht1)
    wd2 = int(s * wd1)

    intrinsics[:, 0] *= float(wd2) / wd1
    intrinsics[:, 1] *= float(ht2) / ht1

    depths = depths.unsqueeze(1)
    depths = F.interpolate(depths, [ht2, wd2], mode='nearest')
    images = F.interpolate(images, [ht2, wd2], mode='bilinear', align_corners=True)

    x0 = np.random.randint(0, wd2-crop_size[1]+1)
    y0 = np.random.randint(0, ht2-crop_size[0]+1)
    x1 = x0 + crop_size[1]
    y1 = y0 + crop_size[0]

    images = images[:, :, y0:y1, x0:x1]
    depths = depths[:, :, y0:y1, x0:x1]
    depths = depths.squeeze(1)

    intrinsics[:, 0, 2] -= x0
    intrinsics[:, 1, 2] -= y0

    return images, depths, intrinsics


def load_pair(file: str):
    with open(file) as f:
        lines = f.readlines()
    n_cam = int(lines[0])
    pairs = {}
    img_ids = []
    for i in range(1, 1+2*n_cam, 2):
        pair = []
        score = []
        img_id = int(lines[i].strip())
        pair_str = lines[i+1].strip().split(' ')
        n_pair = int(pair_str[0])
        for j in range(1, 1+2*n_pair, 2):
            pair.append(int(pair_str[j]))
            score.append(float(pair_str[j+1]))
        img_ids.append(img_id)
        pairs[img_id] = {'id': img_id, 'index': i//2, 'pair': pair, 'score': score}
    pairs['id_list'] = img_ids
    return pairs


def scale_operation(images, intrinsics, s):
    ht1 = images.shape[2]
    wd1 = images.shape[3]
    ht2 = int(s * ht1)
    wd2 = int(s * wd1)
    intrinsics[:, 0] *= s
    intrinsics[:, 1] *= s
    images = F.interpolate(images, [ht2, wd2], mode='bilinear', align_corners=True)
    return images, intrinsics


def crop_operation(images, intrinsics, crop_h, crop_w):
    ht1 = images.shape[2]
    wd1 = images.shape[3]
    x0 = (wd1-crop_w) // 2
    y0 = (ht1-crop_h) // 2
    x1 = x0 + crop_w
    y1 = y0 + crop_h
    images = images[:, :, y0:y1, x0:x1]
    intrinsics[:, 0, 2] -= x0
    intrinsics[:, 1, 2] -= y0
    return images, intrinsics
