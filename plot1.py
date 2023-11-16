import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import itertools
import h5py
from scipy.interpolate import CubicSpline

import h5py
import numpy as np
from scipy.ndimage import convolve1d
from scipy.interpolate import interp1d

bin_width_s=.05
preprocess=True

filename = 'indy_20160630_01.mat'

with h5py.File(filename, "r") as f:
    n_channels = f['chan_names'].shape[1]
    chan_names = []
    for i in range(n_channels):
        chan_names.append(f[f['chan_names'][0, i]][()].tobytes()[::2].decode())
    M1_indices = [i for i in range(n_channels) if chan_names[i].split(' ')[0] == 'M1']
    S1_indices = [i for i in range(n_channels) if chan_names[i].split(' ')[0] == 'S1']
    t = f['t'][0, :]
    result = {}
    for indices in (M1_indices, S1_indices):
        if len(indices) == 0:
            continue
        # Get region (M1 or S1)
        region = chan_names[indices[0]].split(" ")[0]
        # Perform binning
        n_channels = len(indices)
        n_sorted_units = f["spikes"].shape[0] - 1  # The FIRST one is the 'hash' -- ignore!
        d = n_channels * n_sorted_units
        max_t = t[-1]
        n_bins = int(np.floor((max_t - t[0]) / bin_width_s))
        binned_spikes = np.zeros((n_bins, d), dtype=np.int)

        cursor_pos = f['cursor_pos']
        sp = f["spikes"]
        print(len(cursor_pos[0]))
        print(len(sp[0]))
        for chan_idx in indices:
            for unit_idx in range(1, n_sorted_units):  # ignore hash!
                spike_times = f[f["spikes"][unit_idx, chan_idx]][()]
                if spike_times.shape == (2,):
                    # ignore this case (no data)
                    continue
                spike_times = spike_times[0, :]
                # get rid of extraneous t vals
                spike_times = spike_times[spike_times - t[0] < n_bins * bin_width_s]
                bins = np.arange(spike_times[0], t[0] + (n_bins+1) * bin_width_s, bin_width_s)

                break
            break
        break

spike_times = spike_times[0:100]
# bins = bins[0:1000]
spike_counts, _ = np.histogram(spike_times, bins=bins)
# 绘制Raster图
fig, ax = plt.subplots(figsize=(10, 8))
for i, spikes in enumerate(spike_times):
    y = np.ones_like(spikes) * i
    ax.scatter(spikes, y, marker='|', color='k')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Neuron')
plt.savefig('raster.png')


# 绘制PSTH图
fig, ax = plt.subplots()
psth = np.mean(spike_counts, axis=0) / bin_width_s
ax.bar(bins[:-1], psth, width=bin_width_s, align='edge')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Firing rate (Hz)')
plt.savefig('PSTH.png')

