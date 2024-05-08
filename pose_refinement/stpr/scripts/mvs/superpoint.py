from pathlib import Path
import torch
from torch import _nnpack_available

def simple_nms(scores, nms_radius: int):
    assert(nms_radius >= 0) 

    def max_pool(x):
      return torch.nn.functional.max_pool2d(
        x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius
      )
    
    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
      supp_mask = max_pool(max_mask.float()) > 0
      supp_scores = torch.where(supp_mask, zeros, scores)
      new_max_mask = supp_scores == max_pool(supp_scores)
      max_mask = max_mask | (new_max_mask & (~supp_mask))

    return torch.where(max_mask, scores, zeros)

def remove_borders(keypoints, scores, border: int, height: int, width: int):
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]

def top_k_keypoints(keypoints, scores, k:int):
    if k>= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores

def sample_descriptors(keypoints, descriptors, s: int = 8):
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints[None])
    keypoints = keypoints*2 - 1
    args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
    descriptors = torch.nn.functional.grid_sample(
      descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
      descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


class SuperPoint(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """

    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }


    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, self.config['descriptor_dim'], kernel_size=1, stride=1, padding=0)
        print("descriptor dims = ", self.config["descriptor_dim"])

        path = Path(__file__).parent / self.config['path']
        print(path)
        self.load_state_dict(torch.load(str(path)))

        mk = self.config['max_keypoints']
        print("max keypoints = ", mk)
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be postive or \" -1\"')
        print('Loaded SuperPoint model')

    def forward(self, data):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
            x: Image pytorch tensor shaped N x 1 x H x W.
        Output
            semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
            desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Shared Encoder.
        x = self.relu(self.conv1a(data['image']))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        scores = simple_nms(scores, self.config['nms_radius'])

        #extract keypoints
        keypoints = [
            torch.nonzero(s > self.config['keypoint_threshold'])
            for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], h*8, w*8)
            for k, s in zip(keypoints, scores)
        ]))

        #keep the k keypoints with highese score
        if self.config['max_keypoints'] >= 0:
            keypoint, scores = list(zip(*[
                top_k_keypoints(k, s, self.config['max_keypoints'])
                for k, s in zip(keypoints, scores)]))
            
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        descriptors = [sample_descriptors(k[None], d[None], 8)[0]
                       for k,d in zip(keypoints, descriptors)]
        
        return {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
        }
