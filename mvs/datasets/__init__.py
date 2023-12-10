import gin
from torch.utils.data import DataLoader

from .blended import Blended
from .custom import Custom
from .dtu import DTU, DTUTest
from .tnt import TNT
from .waymo import Waymo

dataset_dict = {
    "DTU": DTU,
    "DTUTest": DTUTest,
    "Blended": Blended,
    "TNT": TNT,
    "Custom": Custom,
    "Waymo": Waymo
}

@gin.configurable()
def get_test_data_loader(
    datasetname,
    num_frames=10,
    subset=None,
    **args,
):
    if subset is not None:
        start, end, step = subset
        subset = list(range(start, end, step))
    
    dataset = dataset_dict[datasetname](
        num_frames=num_frames,
        subset=subset,
        **args,
    )
    gpuargs = {'num_workers': 4, 'drop_last' : False, 'shuffle': False, 'pin_memory': True}
    data_loader = DataLoader(dataset, batch_size=1, **gpuargs)
    return data_loader

@gin.configurable()
def get_train_data_loader(
    datasetname,
    batch_size,
    num_frames=10,
):
    dataset = dataset_dict[datasetname](num_frames=num_frames)
    gpuargs = {'num_workers': 4, 'drop_last' : True, 'shuffle': True, 'pin_memory': True}
    data_loader = DataLoader(dataset, batch_size=batch_size, **gpuargs)
    return data_loader

