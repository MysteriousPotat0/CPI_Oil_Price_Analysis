import numpy as np
import keras.metrics
import tensorflow as tf


# Function to calculate different metrics
def metrics(truth_label, prediction_label):
    """
    Takes in prediction label from a model and ground truth label to calculates MSE, MAE, and RMSE
    """

    truth_label = tf.cast(truth_label, dtype=tf.float32)
    prediction_label = tf.cast(prediction_label, dtype=tf.float32)

    mae = keras.metrics.mean_absolute_error(truth_label, prediction_label)
    print(mae.ndim)
    mse = keras.metrics.mean_squared_error(truth_label, prediction_label)
    rmse = tf.sqrt(mse)

    # If prediction data is multivariate
    if mae.ndim >= 1:
        mae = tf.reduce_mean(mae)
        mse = tf.reduce_mean(mse)
        rmse = tf.reduce_mean(rmse)

    return {'mae': mae.numpy(),
            'mse': mse.numpy(),
            'rmse': rmse.numpy()}


# Splitting sequence into expanding window style data
def sliding_window(data, label, window, horizon):
    """
    Split a dataframe into multiple steps with varying window sizes
    :param data: Input sequences
    :param label: Input target data
    :param window: How far is the window size
    :param horizon: How far is the prediction
    :return: Array of training data and label
    """

    x, y = list(), list()

    # Loops through every example in the training data input
    for i in range(len(data)):

        # Find the end of each sequence
        end_ix = i + window
        steps_out = end_ix + horizon

        # Check if it is the end of the sequence

        if end_ix >= len(data):
            break

        # If the end_ix is not at the end of the sequence data
        else:
            print(label[end_ix:steps_out])
            seq_x, seq_y = data[i:end_ix, :], label[end_ix:steps_out]

            if len(seq_y) < horizon:

                seq_y = np.insert(seq_y, 0, seq_y[-1])

            x.append(seq_x)
            y.append(seq_y)

    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


# Function to scale the data back
def rescale_data(data, Scaler, model=None):
    """
    Takes in a 1D array and scale the number back to the original form
    """

    if model is not None:
        train_new = np.zeros(shape=(len(model.predict(data)), 33))
        train_new[:, 0:1] = model.predict(data)[:, 0:1]
        train_new[:, 0:1] = Scaler.inverse_transform(train_new)[:, 0:1]
        print(f'Rescaled Value: {train_new[:, 0:1]}')

    else:
        train_new = np.zeros(shape=(len(data), 33))
        train_new[:, 0:1] = data[0:]
        train_new[:, 0:1] = Scaler.inverse_transform(train_new)[:, 0:1]
        print(f'Rescaled Value: {train_new[:, 0:1]}')

    return train_new[:, 0]
