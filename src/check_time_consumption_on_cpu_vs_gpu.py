import torch
import numpy as np
import time
from tqdm.auto import tqdm
import finite_differences_2d as fd

################################################################################
path_results = '../results/'

# check that a gpu is available
assert torch.cuda.is_available(), 'No gpu is available. Hence the experiment cannot be performed'

# parameters to time
devices = ['cpu', 'cuda:0',]
resolutions = [10, 1000, 5000, 10000, 15000, 20000]

def experiment(device, resolution, n_repetitions=100):
    """
    Computes the first-order finite derivate for a torch.Tensor n_repetitions 
    times while measuring the time consumption.

    Parameters
    ----------
    device : str
        Name of device on which the computations will be performed.
    resolution : int
        Resolution along each axes of the tensor upon which the computations will 
        be performed.
    n_repetitions : int
        Number of times the computation will be performed.

    Returns
    -------
    time_consumption : float
        The time it took to perform the computations.
    """

    # define a torch.Tensor to operate on
    a = torch.rand(resolution, resolution) #define some torch.Tensor
    a = a.to(device) #move the torch.Tensor to the device of interest
    
    # do repeated finite difference computations and time it
    time_0 = time.time()
    
    for i in tqdm(range(n_repetitions)):
    
        fd.dxdx_centered(a)

    time_consumption = time.time() - time_0

    return time_consumption    

# perform timing for each device for each resolution
for device in devices:

    time_consumptions = []

    for resolution in resolutions:
    
        time_consumption = experiment(device, resolution)
        time_consumptions.append(time_consumption)

    # save time_consumptions
    with open(f'{path_results}time_consumptions_{device}.npy', 'wb') as f:
        np.save(f, time_consumptions)

# save resolutions
with open(f'{path_results}resolutions.npy', 'wb') as f:
    np.save(f, resolutions)
