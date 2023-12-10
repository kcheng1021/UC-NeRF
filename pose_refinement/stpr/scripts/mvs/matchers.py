import torch

def mutual_nn_matcher(descriptors1, descriptors2, device='cuda'):
    des1 = torch.from_numpy(descriptors1).to(device)
    des2 = torch.from_numpy(descriptors2).to(device)
    sim = des1 @ des2.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = ids1 == nn21[nn12]
    matches = torch.stack([ids1[mask], nn12[mask]]).t()

    return matches.data.cpu().numpy()

def ratio_matcher(descriptors1, descriptors2, ratio=0.8, device="cuda"):
    des1 = torch.from_numpy(descriptors1).to(device)
    des2 = torch.from_numpy(descriptors2).to(device)
    sim = des1 @ des2.t()

    nns_sim, nns = torch.topk(sim, 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    ratios12 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    nn12 = nns[:, 0]

    nns_sim, nns = torch.topk(sim.t(), 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    ratios21 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    nn21 = nns[:, 0]

    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = torch.min(ratios12 <= ratio, ratios21[nn12] <= ratio)

    matches = torch.stack([ids1[mask], nn12[mask]], dim=-1)

    return matches.data.cpu().numpy()

def mutual_nn_ratio_matcher(descriptors1, descriptors2, ratio=0.8, device="cuda"):
    des1 = torch.from_numpy(descriptors1).to(device)
    des2 = torch.from_numpy(descriptors2).to(device)
    sim = des1 @ des2.t()

    nns_sim, nns = torch.topk(sim, 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    ratios12 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    nn12 = nns[:, 0]

    nns_sim, nns = torch.topk(sim.t(), 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    ratios21 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    nn21 = nns[:, 0]

    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = torch.min(ids1 == nn21[nn12], torch.min(ratios12 <= ratio, ratios21[nn12] <= ratio))

    matches = torch.stack([ids1[mask], nn12[mask]], dim=-1)

    return matches.data.cpu().numpy()