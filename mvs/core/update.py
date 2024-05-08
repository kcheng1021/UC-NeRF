from fastcore.all import store_attr
import torch
import torch.nn as nn
import torch.nn.functional as F
import gin


@gin.configurable()
class ConvGRU(nn.Module):
    def __init__(self, kernel_z=3, kernel_r=3, kernel_q=3, h_planes=None, i_planes=None):
        super(ConvGRU, self).__init__()
        self.do_checkpoint = False
        self.convz = nn.Conv2d(h_planes+i_planes, h_planes, kernel_z, padding=kernel_z//2)
        self.convr = nn.Conv2d(h_planes+i_planes, h_planes, kernel_r, padding=kernel_r//2)
        self.convq = nn.Conv2d(h_planes+i_planes, h_planes, kernel_q, padding=kernel_q//2)

    def forward(self, net, *inputs):
        inp = torch.cat(inputs, dim=1)
        net_inp = torch.cat([net, inp], dim=1)
        z = self.convz(net_inp)
        z = torch.sigmoid(z)
        r = torch.sigmoid(self.convr(net_inp))
        q = torch.tanh(self.convq(torch.cat([r*net, inp], dim=1)))
        net = (1-z) * net + z * q
        return net


@gin.configurable()
class UpdateBlock(nn.Module):
    def __init__(self,
            kernel_corr=3,
            dim0_corr=64,
            dim1_corr=64,
            dim_net=None, # to be filled with that from RAFT
            dim_inp=None,
            dim0_delta=256,
            kernel0_delta=3,
            kernel1_delta=3,
            num_levels=3,
            radius=5,
            size_disp_enc=7,
            kernel0_vis=3,
            kernel1_vis=3,
            
            # whether share some weights
            share_corr=True,
            share_gru=True,
            share_delta=False,

            # default aggregation is mean, which is proved to be better
            aggregation=["mean"],
            cascade=None
        ):
        super(UpdateBlock, self).__init__()
        store_attr()
        cor_planes = len(aggregation) * num_levels * (2 * radius + 1)
        n_cascade = len(cascade)

        for i in (range(n_cascade) if not share_corr else [""]):
            # setattr to let pytorch recognize it is tranable
            setattr(self, f'corr_encoder{i}', nn.Sequential(
                nn.Conv2d(cor_planes, dim0_corr, 1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim0_corr, dim1_corr, kernel_corr, padding=kernel_corr//2),
                nn.ReLU(inplace=True)))

        for i in (range(n_cascade) if not share_delta else [""]):
            setattr(self, f'delta{i}', nn.Sequential(
                nn.Conv2d(dim_net, dim0_delta, kernel0_delta, padding=kernel0_delta//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim0_delta, 1, kernel1_delta, padding=kernel1_delta//2)))

        i_planes = dim_inp + dim1_corr + self.size_disp_enc ** 2
        h_planes = dim_net

        for i in (range(n_cascade) if not share_gru else [""]):
            setattr(self, f'gru{i}', ConvGRU(h_planes=h_planes, i_planes=i_planes))


    def disp_encoder(self, disp):
        batch, _, ht, wd = disp.shape
        dispkxk = F.unfold(disp, [self.size_disp_enc,self.size_disp_enc], padding=self.size_disp_enc//2)
        dispkxk = dispkxk.view(batch, self.size_disp_enc ** 2, ht, wd)
        disp1x1 = disp.view(batch, 1, ht, wd)
        return dispkxk - disp1x1

    def forward(self, net, inp, disp, corr_frames, stage):

        batch, num, ch, ht, wd = net.shape
        inp_shape = (batch*num, -1, ht, wd)
        out_shape = (batch, num, -1, ht, wd)

        net = net.view(*inp_shape)
        inp = inp.view(*inp_shape)
        disp = disp.view(*inp_shape)

        disp = 100 * self.disp_encoder(disp)
        
        batch, n_view, ch, ht, wd = corr_frames.shape

        corr_parts = []
        if "mean" in self.aggregation:
            corr_parts.append(torch.mean(corr_frames, dim=1))
        if "max" in self.aggregation:
            corr_parts.append(torch.max(corr_frames, dim=1).values)
        if "std" in self.aggregation:
            corr_parts.append(torch.std(corr_frames, dim=1))

        corr = torch.stack(corr_parts, dim=2)
        corr = corr.view(*inp_shape)
        corr = getattr(self, f'corr_encoder{stage if not self.share_corr else ""}')(corr)
        net = getattr(self, f'gru{stage if not self.share_gru else ""}')(net, inp, disp, corr)

        delta = .01 * getattr(self, f'delta{stage if not self.share_delta else ""}')(net)

        net = net.view(*out_shape)
        delta = delta.view(*out_shape)
        delta = delta.squeeze(2)

        return net, delta

