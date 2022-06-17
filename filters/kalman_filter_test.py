import numpy as np
from kalman_filter import KalmanFilter

F = np.array([(1, 0, 1, 0, 0, 0),
              (0, 1, 0, 1, 0, 0),
              (0, 0, 1, 0, 1, 0),
              (0, 0, 0, 1, 0, 1),
              (0, 0, 0, 0, 1, 0),
              (0, 0, 0, 0, 0, 1)])

P = np.array([(4, 0, 0, 0, 0, 0),
              (0, 4, 0, 0, 0, 0),
              (0, 0, 2, 0, 0, 0),
              (0, 0, 0, 2, 0, 0),
              (0, 0, 0, 0, 1, 0),
              (0, 0, 0, 0, 0, 1)])

Q = np.array([(1, 0, 0, 0, 0, 0),
              (0, 1, 0, 0, 0, 0),
              (0, 0, 1, 0, 0, 0),
              (0, 0, 0, 1, 0, 0),
              (0, 0, 0, 0, 1, 0),
              (0, 0, 0, 0, 0, 1)])

H = np.array([(1, 0, 0, 0, 0, 0),
              (0, 1, 0, 0, 0, 0)])

R = np.array([(2, 0),
              (0, 2)])

# Create Kalman Filter object
kf = KalmanFilter(F, P, Q, H, R)

# Input data
observations = np.array([(0.8, 1.91),
                         (1.6, 3.82),
                         (2.4, 5.64),
                         (3.2, 7.37),
                         (4.0, 9.01),
                         (4.8, 10.56),
                         (5.6, 12.02),
                         (6.4, 13.39),
                         (7.2, 14.67),
                         (8.0, 15.86),
                         (8.8, 16.96),
                         (9.6, 17.97),
                         (10.4, 18.89),
                         (11.2, 19.72),
                         (12.0, 20.46),
                         (12.8, 21.11),
                         (13.6, 21.67),
                         (14.4, 22.14),
                         (15.2, 22.52),
                         (16.0, 22.81)])

# What the predictions should be - used to check our results
given_predictions = np.array([(0.622222222222, 1.48555555556),
                              (1.42222222222, 3.39555555556),
                              (2.34336283186, 5.51911504425),
                              (3.21666666667, 7.42656053459),
                              (4.03265874051, 9.0998649572),
                              (4.8236779076, 10.6226863497),
                              (5.61104139065, 12.0485689789),
                              (6.40268033352, 13.3966755652),
                              (7.19914880075, 14.6676441356),
                              (7.99853762498, 15.8561905767),
                              (8.79902509173, 16.9575129271),
                              (9.59960641694, 17.9690245406),
                              (10.3999531269, 18.8899071281),
                              (11.200076077, 19.7202094593),
                              (12.0000786684, 20.4602064206),
                              (12.8000447832, 21.1101148307),
                              (13.6000151777, 21.6700376364),
                              (14.3999998854, 22.1399986579),
                              (15.1999955106, 22.5199879892),
                              (15.9999962388, 22.8099902445)])

# Array of our KF predictions; we will use np.allclose to compare these with the predictions listed directly above
predictions = []

for observation in observations:
    corrected_prediction = kf.make_corrected_prediction(observation.reshape(-1, 1))  # Make our KF prediction
    observed_x, observed_y = observation  # Extract the observed x and y (for the print statement)
    pred_x, pred_y = corrected_prediction[0][0], corrected_prediction[1][0]  # Extract the predicted x and y
    predictions.append((pred_x, pred_y))  # Store the predictions

    print(f'observed x = {observed_x}, observed y = {observed_y}, KF x prediction = {pred_x}, '
          f'KF y prediction = {pred_y}')  # Visual check to show that our version of KF works

# Computational check to show that our version of KF works
preds_close_to_given_preds = np.allclose(given_predictions, np.array(predictions))

print(f'\n----------------------------------------------------------------------\nOur predictions are extremely close '
      f'to the given predictions (should be "True" if our implementation is correct: {preds_close_to_given_preds}')
