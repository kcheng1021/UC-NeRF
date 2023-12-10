from fastcore.all import store_attr
import torch
import torch.nn as nn
from core.extractor import BasicEncoder
from core.corr import CorrBlock
from core.update import UpdateBlock
import gin

autocast = torch.cuda.amp.autocast


@gin.configurable()
class RAFT(nn.Module):
    def __init__(self,
            # each tuple is a cascade stage, (D, N, T), with D hypotheses, (-1 means auto according to downsample pyramid), and each interval of size 1 / 400 / N, iteration number is T
            cascade=[(64, 64, 8), (-1, 320, 8)],
            # default is "HR", from H x W to H / 4 to W / 4, another option os "LR", from H x W to H / 8 to W / 8
            encoder_type="HR",
            dim_fmap=64,
            dim_net=64,
            dim_inp=64,
            test_mode=False,
        ):
        super(RAFT, self).__init__()
        store_attr()
        self.dim_net = dim_net
        self.dim_inp = dim_inp
        
        self.fnet = BasicEncoder(output_dim=dim_fmap, norm_fn='instance', type=encoder_type)
        self.cnet = BasicEncoder(output_dim=dim_net+dim_inp, norm_fn='none', type=encoder_type)
        self.update_block = UpdateBlock(cascade=cascade, dim_net=dim_net, dim_inp=dim_inp)

    @gin.configurable()
    def forward(self, images, poses, intrinsics, scale=None, do_report=False):
        if scale is not None: poses[..., :3, 3] *= scale
        test_mode = self.test_mode
        intrinsics = intrinsics.clone()
        factor = 8 if self.encoder_type == "LR" else 4
        intrinsics[:, :, :2] /=  factor
        images *= 2 / 255. 
        images -= 1
        batch, num, ch, ht, wd = images.shape

        # ii and jj means image pairs, it was designed to be extendable, but now ii are all 0, which is the reference image
        ii = [0] * (num - 1)
        jj = list(range(1, num))
        ii = torch.as_tensor(ii).to(images.device)
        jj = torch.as_tensor(jj).to(images.device)
        ht //= factor
        wd //= factor
        
        disp = torch.zeros(batch, 1, ht, wd).to(images.device).float()


        with autocast(enabled=True):
            
            net_inp = self.cnet(images[:,[0]])
            net, inp = net_inp.split([self.dim_net, self.dim_inp], dim=2)
            net = torch.tanh(net)
            inp = torch.relu(inp)

            # different strategy in test time to save memory
            if not test_mode:
                fmaps = self.fnet(images)
            else:
                fmaps = []
                for i in range(num):
                    fmaps.append(self.fnet(images[:, [i]]))
                fmaps = torch.cat(fmaps, 1)

            if test_mode: del images

            predictions = []

            stage = 0
            for nIncre, incre, nIters in self.cascade:
                if nIncre == -1:
                    radius = self.update_block.radius
                    num_levels = self.update_block.num_levels
                    nIncre = (2 * radius + 1) * 2 ** (num_levels - 1)
                incre = 0.0025 / incre
                    
                with autocast(enabled=False):
                    corr_fn = CorrBlock(
                        fmaps, poses, intrinsics, ii, jj,
                        nIncre=nIncre, incre=incre, disps_input=disp.detach(),
                        # shift center of depth hypotheses away from current estimation (0) in stage 0;
                        # otherwise hypotheses are centered around current estimation
                        shift=stage==0,
                        num_levels=self.update_block.num_levels,
                        radius=self.update_block.radius,
                        test_mode=test_mode,
                        do_report=do_report
                    )
                
                for itr in range(nIters):
                    disp = disp.detach()
                    with autocast(enabled=False):
                        corr_frames = corr_fn(disp[:,ii])
                    net, delta = self.update_block(net, inp, disp, corr_frames, stage)
                    disp = disp + delta.float()

                    if not test_mode: predictions.append(disp)

                stage += 1
        if test_mode:
            assert(scale is not None)
            return disp * scale.cuda()
        return predictions