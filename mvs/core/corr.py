import torch
import torch.nn.functional as F
import alt_cuda_corr
import torch
import gin
from utils.memory import report
from utils.projective_ops import projective_transform
from utils.bilinear_sampler import bilinear_sampler1


# Inherit from Function
class DirectCorr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fmap1, fmap2, coords):
        ctx.save_for_backward(fmap1, fmap2, coords)
        corr, = alt_cuda_corr.forward(fmap1, fmap2, coords, 0)
        return corr

    def backward(ctx, grad_output):
        fmap1, fmap2, coords = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        fmap1_grad, fmap2_grad, coords_grad = \
            alt_cuda_corr.backward(fmap1, fmap2, coords, grad_output, 0)

        return fmap1_grad, fmap2_grad, coords_grad


def direct_corr(fmaps, x1, ii, jj, DD):
    fmaps = fmaps.permute(0,1,3,4,2)
    fmaps1 = fmaps[:,ii] / 8.0
    fmaps2 = fmaps[:,jj] / 8.0

    batch, num, h1, w1, ch = fmaps1.shape
    fmaps1 = fmaps1.reshape(batch*num, h1, w1, ch).contiguous().float()
    fmaps2 = fmaps2.reshape(batch*num, h1, w1, ch).contiguous().float()

    x1 = x1.reshape(batch*num, h1, w1, -1, 2)
    x1 = x1.permute(0,3,1,2,4).contiguous()

    corr = DirectCorr.apply(fmaps1, fmaps2, x1)
    corr = corr.permute(0,2,3,4,1)

    return corr.reshape(batch*num*h1*w1, 1, 1, DD)

class CorrBlock:
    def __init__(self, fmaps, poses, intrinsics, ii, jj, nIncre, incre, disps_input, shift, num_levels, radius, test_mode, do_report):
        self.num_levels = num_levels
        self.radius = radius
        self.test_mode = test_mode
        self.nIncre = nIncre
        self.incre = incre
        device = fmaps.device
        fmaps = fmaps.float()
        batch, num_frames, ch, h1, w1 = fmaps.shape
        opt_num = 1
        disps = ((torch.arange(nIncre) - nIncre // 2) * incre).to(device).view(1, 1, nIncre, 1, 1)
        disps_input = disps_input.view(batch, opt_num, 1, h1, w1)

        if shift:
            self.disps_origin = torch.where(disps_input < nIncre // 2 * incre, torch.tensor(nIncre // 2 * incre).cuda().float(), disps_input)
        else:
            self.disps_origin = disps_input.clone()


        disps = disps + self.disps_origin
        
        if not self.test_mode:
            num = ii.shape[0]
            segs = list(range(num + 1))
            corr_parts = []
            for j in range(len(segs) - 1):
                cur_num = segs[j + 1] - segs[j]
                x1 = projective_transform(poses, disps, intrinsics, ii[segs[j]:segs[j + 1]], jj[segs[j]:segs[j + 1]])
                x1 = x1[..., [0,1]].permute(0,1,3,4,2,5).contiguous()
                x1 = x1.clamp(min=-1e4, max=1e4)
                corr_parts.append(direct_corr(fmaps, x1, ii[segs[j]:segs[j + 1]], jj[segs[j]:segs[j + 1]], nIncre).view(batch, cur_num, h1*w1, 1, 1, nIncre))
            corr = torch.cat(corr_parts, dim=1)
            corr = corr.reshape(-1, 1, 1, nIncre)

        else:
            num = ii.shape[0]
            segs = list(range(num + 1))
            corr_parts = []
            for j in range(len(segs) - 1):
                cur_num = segs[j + 1] - segs[j]
                x1 = projective_transform(poses, disps, intrinsics, ii[segs[j]:segs[j + 1]], jj[segs[j]:segs[j + 1]])
                x1 = x1[..., [0,1]].permute(0,1,3,4,2,5).contiguous()
                x1 = x1.clamp(min=-1e4, max=1e4)
                corr_parts.append(direct_corr(fmaps, x1, ii[segs[j]:segs[j + 1]], jj[segs[j]:segs[j + 1]], nIncre).view(batch, cur_num, h1*w1, 1, 1, nIncre))
            corr = torch.cat(corr_parts, dim=1)
            corr = corr.reshape(-1, 1, 1, nIncre)
        
        
        self.corr_pyramid = [ corr ]
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, [1,2], stride=[1,2])
            self.corr_pyramid.append(corr)
            # HERE is the peak memory usage check point
            if do_report and i == self.num_levels-2 and not shift: report()


    def __call__(self, zinv):
        r = self.radius
        batch, num, h1, w1 = zinv.shape
        zinv = zinv.view(batch*num, h1, w1, 1)

        coords = torch.maximum((zinv - self.disps_origin.view(batch, h1, w1, 1).repeat(num, 1, 1, 1)) / self.incre + self.nIncre // 2, torch.Tensor([0]).to(zinv.device)) # 0

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            if not self.test_mode:
                dx = torch.linspace(-r, r, 2*r+1)
                dx = dx.view(1, 1, 2*r+1, 1).to(coords.device)
                x0 = dx + coords.reshape(batch*num*h1*w1, 1, 1, 1) / 2**i
                y0 = torch.zeros_like(x0)
                coords_lvl = torch.cat([x0,y0], dim=-1)
                m = 1
                coords_lvl = coords_lvl.repeat(m, 1, 1, 1)
                corr = bilinear_sampler1(corr, coords_lvl)
                corr = corr.view(m, batch*num, h1, w1, -1).permute(1, 2, 3, 0, 4).reshape(batch*num, h1, w1, -1)

            else:
                segs = list(range(-r, r + 1))
                corr_parts = []
                for j in range(len(segs)):
                    dx = torch.tensor([segs[j]])
                    dx = dx.view(1, 1, 1, 1).to(coords.device)
                    x0 = dx + coords.reshape(batch*num*h1*w1, 1, 1, 1) / 2**i
                    y0 = torch.zeros_like(x0)
                    coords_lvl = torch.cat([x0,y0], dim=-1)
                    sub_parts = []
                    chunk_num = 1
                    assert(batch * h1 * w1 * num % chunk_num == 0)
                    chunk_size = batch * h1 * w1 * num // chunk_num
                    for k in range(chunk_num):
                        sub_parts.append(bilinear_sampler1(corr[chunk_size * k: chunk_size * (k + 1)], coords_lvl[chunk_size * k: chunk_size * (k + 1)]))
                    corr_parts.append(torch.cat(sub_parts, 0).view(batch*num, h1, w1, -1))
                corr = torch.cat(corr_parts, dim=-1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1).permute(0, 3, 1, 2)
        return out.reshape(batch, num, -1, h1, w1).contiguous()




    @staticmethod
    def corr(fmaps, ii, jj):
        fmap1 = fmaps[:, ii]
        fmap2 = fmaps[:, jj]

        batch, num, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.reshape(batch*num, dim, ht*wd) / 8.0
        fmap2 = fmap2.reshape(batch*num, dim, ht*wd) / 8.0
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        return corr.view(batch, num, ht, wd, 1, ht, wd)