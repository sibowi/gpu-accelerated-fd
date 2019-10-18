import matplotlib.pyplot as plt
import numpy as np

path_results = '../results/'
path_figures = '../figures/'

# load data
time_consumptions_cpu = np.load(f'{path_results}time_consumptions_cpu.npy')
time_consumptions_gpu = np.load(f'{path_results}time_consumptions_cuda:0.npy')
resolutions = np.load(f'{path_results}resolutions.npy')

# create figure
fig, axs = plt.subplots(1, 1, figsize=(10,5))
axs.plot(resolutions, time_consumptions_cpu, marker='.', ms=14, ls='--',
         label='cpu')
axs.plot(resolutions, time_consumptions_gpu, marker='.', ms=14, ls='--',
         label='gpu')
axs.legend()
axs.set_xlabel('Resolution of 2D grid')
axs.set_ylabel('Time to perform 100 operations [s]')
axs.set_xticks(resolutions)
axs.grid(True)

plt.savefig(f'{path_figures}results.png', bbox_inches='tight')
