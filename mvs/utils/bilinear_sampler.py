import torch
import torch.nn.functional as F



def bilinear_sampler1(img, coords):
    """ Wrapper for grid_sample, uses pixel coordinates 
        Because it is 1-D sampler, one dimension is of size 1
    """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)

    b = img.shape[0]
    imgs = []
    n = 16
    assert(b % n == 0)
    chunk_size = b // n
    for i in range(n):
        imgs.append(F.grid_sample(img[i * chunk_size: (i+1) * chunk_size], grid[i * chunk_size: (i+1) * chunk_size], align_corners=True))
    img = torch.cat(imgs, 0)

    return img



def bilinear_sampler(img, coords, mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

