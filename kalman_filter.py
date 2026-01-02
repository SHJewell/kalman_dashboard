
'''
Kalman Filter implementation by Marko Cotra from:
https://medium.com/towards-data-science/wtf-is-sensor-fusion-part-2-the-good-old-kalman-filter-3642f321440
'''

import numpy as np
import time

class MotionModel():
    def __init__(self, A, Q):
        self.A = A
        self.Q = Q

        (m, _) = Q.shape
        self.zero_mean = np.zeros(m)

    def __call__(self, x):
        new_state = self.A @ x + np.random.multivariate_normal(self.zero_mean, self.Q)
        return new_state


class MeasurementModel():
    def __init__(self, H, R):
        self.H = H
        self.R = R

        (n, _) = R.shape
        self.zero_mean = np.zeros(n)

    def __call__(self, x):
        measurement = self.H @ x + np.random.multivariate_normal(self.zero_mean, self.R)
        return measurement


class system_sim():

    def __init__(self, s2_x=0.1, s2_y=0.1, T=1, lambda2=0.3):
        self.s2_x = s2_x ** 2
        self.s2_y = s2_y ** 2
        self.T = T
        self.lambda2 = lambda2 ** 2

    def create_model_parameters(self):

        # Motion model parameters
        F = np.array([[1, self.T],
                      [0, 1]])
        base_sigma = np.array([[self.T ** 3 / 3, self.T ** 2 / 2],
                               [self.T ** 2 / 2, self.T]])

        sigma_x = self.s2_x * base_sigma
        sigma_y = self.s2_y * base_sigma

        zeros_2 = np.zeros((2, 2))
        A = np.block([[F, zeros_2],
                      [zeros_2, F]])
        Q = np.block([[sigma_x, zeros_2],
                      [zeros_2, sigma_y]])

        # Measurement model parameters
        H = np.array([[1, 0, 0, 0],
                      [0, 0, 1, 0]])
        R = self.lambda2 * np.eye(2)

        return A, H, Q, R


    def simulate_system(self, K, x0):
        (A, H, Q, R) = self.create_model_parameters()

        # Create models
        motion_model = MotionModel(A, Q)
        meas_model = MeasurementModel(H, R)

        (m, _) = Q.shape
        (n, _) = R.shape

        state = np.zeros((K, m))
        meas = np.zeros((K, n))

        # initial state
        x = x0
        for k in range(K):
            x = motion_model(x)
            z = meas_model(x)

            state[k, :] = x
            meas[k, :] = z

        return state, meas


class KalmanFilter():
    def __init__(self, A, H, Q, R, x_0, P_0):
        # Model parameters
        self.A = A
        self.H = H
        self.Q = Q
        self.R = R

        # Initial state
        self._x = x_0
        self._P = P_0

    def predict(self):
        self._x = self.A @ self._x
        self._P = self.A @ self._P @ self.A.transpose() + self.Q

    def update(self, z):
        self.S = self.H @ self._P @ self.H.transpose() + self.R
        self.V = z - self.H @ self._x
        self.K = self._P @ self.H.transpose() @ np.linalg.inv(self.S)

        self._x = self._x + self.K @ self.V
        self._P = self._P - self.K @ self.S @ self.K.transpose()

    def get_state(self):
        return self._x, self._P

def eval_filter(K, sig_x, sig_y, T=1, lambda2=0.3):

    our_system = system_sim(s2_x=sig_x, s2_y=sig_y, T=T, lambda2=lambda2)

    t = time.time_ns()
    np.random.seed(t % (2**32))

    (A, H, Q, R) = our_system.create_model_parameters()

    # initial state
    x = np.array([0, 0.1, 0, 0.1])
    P = 0 * np.eye(4)

    (state, meas) = our_system.simulate_system(K, x)
    kalman_filter = KalmanFilter(A, H, Q, R, x, P)

    est_state = np.zeros((K, 4))
    est_cov = np.zeros((K, 4, 4))

    for k in range(K):
        kalman_filter.predict()
        kalman_filter.update(meas[k, :])
        (x, P) = kalman_filter.get_state()

        est_state[k, :] = x
        est_cov[k, ...] = P

    return state, est_state, meas
