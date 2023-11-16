import h5py
import numpy as np
from scipy.ndimage import convolve1d
from scipy.interpolate import interp1d
def sum_over_chunks(X, stride):
    X_trunc = X[:len(X) - (len(X) % stride)]
    reshaped = X_trunc.reshape((len(X_trunc) // stride, stride, X.shape[1]))
    summed = reshaped.sum(axis=1)
    return summed


def moving_center(X, n, axis=0):
    if n % 2 == 0:
        n += 1
    w = -np.ones(n) / n
    w[n // 2] += 1
    X_ctd = convolve1d(X, w, axis=axis)
    return X_ctd


def calc_autocorr_fns(X, T):
    autocorr_fns = np.zeros((X.shape[1], T))
    for dt in range(T):
        autocorr_fns[:, dt] = np.sum((X[dt:] * X[:len(X) - dt]), axis=0) / (len(X) - dt)
    return autocorr_fns

def load_sabes_data(filename, bin_width_s=.05, high_pass=True, sqrt=True, thresh=5000,
                    zscore_pos=True):
    # Load MATLAB file
    with h5py.File(filename, "r") as f:
        # Get channel names (e.g. M1 001 or S1 001)
        n_channels = f['chan_names'].shape[1]
        chan_names = []
        for i in range(n_channels):
            chan_names.append(f[f['chan_names'][0, i]][()].tobytes()[::2].decode())
        # Get M1 and S1 indices
        M1_indices = [i for i in range(n_channels) if chan_names[i].split(' ')[0] == 'M1']
        S1_indices = [i for i in range(n_channels) if chan_names[i].split(' ')[0] == 'S1']
        # Get time
        t = f['t'][0, :]
        # Individually process M1 and S1 indices
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
            for chan_idx in indices:
                for unit_idx in range(1, n_sorted_units):  # ignore hash!
                    spike_times = f[f["spikes"][unit_idx, chan_idx]][()]
                    if spike_times.shape == (2,):
                        # ignore this case (no data)
                        continue
                    spike_times = spike_times[0, :]
                    # get rid of extraneous t vals
                    spike_times = spike_times[spike_times - t[0] < n_bins * bin_width_s]
                    bin_idx = np.floor((spike_times - t[0]) / bin_width_s).astype(np.int)
                    unique_idxs, counts = np.unique(bin_idx, return_counts=True)
                    # make sure to ignore the hash here...
                    binned_spikes[unique_idxs, chan_idx * n_sorted_units + unit_idx - 1] += counts
            binned_spikes = binned_spikes[:, binned_spikes.sum(axis=0) > thresh]
            if sqrt:
                binned_spikes = np.sqrt(binned_spikes)
            if high_pass:
                binned_spikes = moving_center(binned_spikes, n=600)
            result[region] = binned_spikes
        # Get cursor position
        cursor_pos = f["cursor_pos"][:].T
        # Line up the binned spikes with the cursor data
        t_mid_bin = np.arange(len(binned_spikes)) * bin_width_s + bin_width_s / 2
        cursor_pos_interp = interp1d(t - t[0], cursor_pos, axis=0)
        cursor_interp = cursor_pos_interp(t_mid_bin)
        if zscore_pos:
            cursor_interp -= cursor_interp.mean(axis=0, keepdims=True)
            cursor_interp /= cursor_interp.std(axis=0, keepdims=True)
        result["cursor"] = cursor_interp

        wf = f["wf"][2,5]
        result["wf"] = wf
        print(wf)
        return result

if __name__ == '__main__':
    filename = '/home/suhanchen/Brain-Computer Intelligence/HW2/indy_20160630_01.mat'
    res= load_sabes_data(filename)