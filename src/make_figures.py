import matplotlib.pyplot as plt
import numpy as np

path_results = '../results/'

time_consumptions_cpu = np.loadtxt(f'{path_results}time_consumptions_cpu.txt', dtype=list)
time_consumptions_gpu = np.loadtxt(f'{path_results}time_consumptions_cuda:0.txt', dtype=object)
resolutions = np.loadtxt(f'{path_results}resolutions.txt', dtype=object)

print(time_consumptions_cpu)
