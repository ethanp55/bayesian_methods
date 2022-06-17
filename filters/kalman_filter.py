import numpy as np


class KalmanFilter:
    def __init__(self, F, P, Q, H, R):
        # Given values
        self._F = F
        self._P = P
        self._Q = Q
        self._H = H
        self._R = R

        # Initialize the other values we need
        self._n_vars = self._F.shape[0]

        self._x = np.zeros(shape=(self._n_vars, 1))

        self._K = np.zeros(shape=(self._n_vars, self._n_vars))

    def _make_prediction(self):
        # Update the state (self._x) and covariance matrix (self._P)
        self._x = self._F.dot(self._x)
        self._P = self._F.dot(self._P).dot(self._F.T) + self._Q

    def _calculate_optimal_gain(self):
        # Calculate the Kalman gain
        # Put the inverse stuff on its own line to make the code a little neater
        inv_stuff = np.linalg.inv(self._H.dot(self._P).dot(self._H.T) + self._R)

        self._K = self._P.dot(self._H.T).dot(inv_stuff)

    def _apply_correction(self, measured_vals):
        # Use the proper equations to update our predictions using the measured values
        self._x = self._x + self._K.dot(measured_vals - self._H.dot(self._x))
        sub = self._K.dot(self._H)
        self._P = (np.eye(sub.shape[0]) - sub).dot(self._P)

        # Return the final state vector (can pull the needed predictions from indexing into it)
        return self._x

    def make_corrected_prediction(self, measured_vals):
        self._make_prediction()  # Step 1 - make a new prediction
        self._calculate_optimal_gain()  # Step 2 - Update the optimal gain value
        final_prediction = self._apply_correction(measured_vals)  # Step 3 - Use measured values to make a correction

        return final_prediction  # Return the final/corrected prediction
