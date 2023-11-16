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


print(X_m1.shape)

training_size = int(0.8 * np.size(X_m1, 1))
x_training = X_m1[:, :training_size]
z_training = Z_m1[:, :training_size]
x_test = X_m1[:, training_size:]
z_test = Z_m1[:, training_size:]

x_predict = np.zeros_like(x_test)
# Instantiating a Kalman filter object (Inputs are state and neural training data
mykf = mkf(x=x_training, z=z_training)
# in case it's slow to do the whole test data we could change the rng
# rng = int(np.size(x_test, 1))
rng=1000

# Estimating a state with each of neural test data
for i in range(rng):
    mykf.predict()
    z_in = z_test[:, i]
    x_predict[:, i] = np.reshape(mykf.update(z_in), (mykf.m, ))
    if i%50 == 0:
        print('KF - Step: ' + str(i) + ' out of ' + str(rng))


# Plotting the results
plt.subplot(2,1,1)
plt.plot(x_predict[0, :rng], label="KF")
plt.plot(x_test[0, :rng], label="actual")
plt.title("X-Position")

plt.subplot(2,1,2)
plt.plot(x_predict[1, :rng], label="KF")
plt.plot(x_test[1, :rng], label="actual")
plt.title("Y-Position")

# plt.legend()
# plt.subplot(4,1,3)
# plt.plot(x_predict[2, :], label="KF")
# plt.plot(states[2, -rng:], label="actual")
# plt.title("X-Velocity")

# plt.legend()
# plt.subplot(4,1,4)
# plt.plot(x_predict[3, :], label="KF")
# plt.plot(states[3, -rng:], label="actual")
# plt.title("Y-Velocity")
plt.tight_layout()
plt.savefig('pos.jpg')