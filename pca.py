import numpy as np
import numpy as np
import datetime
import pytz
import h5py
import hdf5storage
import os
from pynwb import NWBFile, NWBHDF5IO
from pynwb.device import Device
from pynwb.ecephys import SpikeEventSeries
from pynwb.behavior import Position, SpatialSeries
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
fpath_nwb = 'indy_20160630_01.nwb'
io = NWBHDF5IO(fpath_nwb, mode='r')
nwb= io.read()
data = np.array(nwb.acquisition['M1 Spike Events electrode 0 and unit 0'].data)

data = data[:500,:]

# 创建PCA对象
pca = PCA(n_components=2)

# 对数据进行降维
data_pca = pca.fit_transform(data)

# 绘制散点图
plt.scatter(data_pca[:, 0], data_pca[:, 1])
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.savefig('pca.png')


kmeans = KMeans(n_clusters=3, random_state=0).fit(data_pca)

# 绘制聚类结果
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=kmeans.labels_)
plt.savefig('kmeans.png')