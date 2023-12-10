import torch.nn.functional as F
import gin

@gin.configurable()
def sequence_loss(disp_est, disp_gt,
                        depthloss_threshold=100,
                        gradual_weight=None,
                        gamma=0.9,
                        depth_cut=1e-3):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(disp_est)    
    flow_loss = 0.0

    valid = disp_gt > 0.0
    ht, wd = disp_gt.shape[-2:]

    for i in range(n_predictions):
        disp_est[i] = F.interpolate(disp_est[i], [ht, wd], mode='bilinear', align_corners=True)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        loss_disp = (disp_est[i] - disp_gt).abs()
        loss_depth = (1.0 / disp_est[i].clamp(min=depth_cut) - 1.0 / disp_gt.clamp(min=depth_cut)).abs()
        loss_depth = loss_depth.clamp(max=depthloss_threshold) / 3.6e5
        i_loss = gradual_weight * loss_depth + (1 - gradual_weight) * loss_disp
        flow_loss += i_weight * (valid * i_loss).mean()
        flow_loss += .01 * i_weight * (i_loss).mean()

    epe = (1.0/disp_est[-1].clamp(min=depth_cut) - 1.0/disp_gt).abs()
    epe = epe.view(-1)[valid.view(-1)]


    metrics = {
        'mean_depth_error': epe.mean().item(),
        'less3': (epe < 3).float().mean().item(),
        'less10': (epe < 10).float().mean().item(),
        'less25': (epe < 25).float().mean().item(),
    }

    return flow_loss, metrics

