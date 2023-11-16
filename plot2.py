from data import *
from kalman import ManualKalmanFilter as mkf
import matplotlib.pyplot as plt
from filterpy.kalman.UKF import UnscentedKalmanFilter as ukf
from filterpy.kalman import MerweScaledSigmaPoints

filename = 'indy_20160630_01.mat'
m1 = load_sabes_data(filename)
Z_m1, X_m1 = m1["M1"], m1["cursor"]

X_m1 = X_m1.T

Z_m1 = Z_m1.T

v = np.empty_like(X_m1)
acc = np.empty_like(X_m1)

a= np.diff(X_m1, axis=1)
v[:, 0] = 0
v[:, 1:] = a
X_m1 = np.concatenate((X_m1, v), axis=0)

a = np.diff(a, axis=1)
acc[:, 0:2] = 0
acc[:, 2:] = a
X_m1 = np.concatenate((X_m1, acc), axis=0)

# rng=30
# plt.plot(X_m1[0,:rng], Z_m1[0,:rng], '.')
# plt.xlabel('X_pos')
# plt.ylabel('Neu')
# plt.savefig('curve_pos.png')



# print(X_m1.shape)
# rng=30
# plt.plot(X_m1[2,:rng], Z_m1[2,:rng], '.')
# plt.xlabel('Acceleration')
# plt.ylabel('X_Neu')
# plt.savefig('curve_v.png')

print(X_m1.shape)
rng=30
plt.plot(X_m1[4,:rng], Z_m1[4,:rng], '.')
plt.xlabel('X_Velocity')
plt.ylabel('Neu')
plt.savefig('curve_acc.png')