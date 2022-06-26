"""
Goal: Using oil prices and oil production, try and do forcasting on the CPI value of each month

Model experimenting ideas
1. Naive model
2. Dense model with linear output
3. CNN model
4. RNN model
"""

# Importing libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from Forecasting_Neural_Network import *
from sklearn.preprocessing import MinMaxScaler
tf.random.set_seed(42)

# Import the data from the csv files
CPI_data = pd.read_csv('Data/CPI_data_1997_to_2022.csv')
Energy_Prices = pd.read_csv('Data/Energy_Prices.csv')
Oil_Production = pd.read_csv('Data/International_PetroleumProduction_Consumption_and_Inventories.csv')

# Merge the dataframe into a single dataframe
All_data = CPI_data.merge(Energy_Prices, on='DATE').merge(Oil_Production, on='DATE')

# Turn the date from YY-MM-DD into 1, 2, 3... consecutive months
month_seq = np.arange(1, len(All_data) + 1)
All_data['DATE'] = month_seq


# Calculating the metrics for naive model
naive_predict = naives(All_data['VALUE'].dropna())
naive_metric = metrics(All_data.loc[1:, 'VALUE'], naives(All_data['VALUE']).dropna())
naive_metric

# Plot the auto-correlation of CPI
acf_plot(All_data['VALUE'])

# Min-Max scale all data
Scaler = MinMaxScaler()
scaled_sliding_window_data = pd.DataFrame(Scaler.fit_transform(All_data.iloc[:, 1:]))

# Lag the data by one day
scaled_sliding_window_data = scaled_sliding_window_data.shift(1).dropna()


# 12 Months sliding window data
train, label = sliding_window(scaled_sliding_window_data.iloc[:, 1:].values,
                              scaled_sliding_window_data.iloc[:, 0].values,
                              window=12,
                              horizon=1)

# Split data int 80/20 train test configuration
x_train, val_data, y_label, val_label = train_test_split(train,
                                                         label,
                                                         shuffle=False,
                                                         test_size=0.2)


"""# Creating a tensorboard callback
log_dir = 'logs/fit' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tb_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
"""
# Creating model checkpoint
checkpoint = keras.callbacks.ModelCheckpoint('checkpoint/model',
                                             monitor='val_loss',
                                             mode='auto')

# Use the keras callback for to reduce learning rate
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                              patience=20,
                                              factor=0.5,
                                              min_lr=1e-6)

# Keras call back for stopping early
stop_early = keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=20)

# Creating a dense fully connected that has a linear output
dense_model = dense_linear()

dense_model.compile(loss='mae',
                    optimizer='adam',
                    metrics='mae')

hist_dense = dense_model.fit(x_train,
                             y_label,
                             epochs=200,
                             validation_data=(val_data, val_label),
                             callbacks=[reduce_lr, stop_early])

# Plot the dense model
plot_hist(hist_dense)

# Find the mse, mae, rmse of the dense model
dense_metrics = metrics(val_label, dense_model.predict(val_data))

dense_pred = rescale_data(val_data, Scaler, dense_model)

# Plot the prediction on validation data vs validation label
plot_pred(rescale_data(val_label, Scaler), dense_pred)

"""# Creating a Convoluted Neural Network
cnn_model = CNN_model()
cnn_model.compile(loss='mae',
                  optimizer='adam',
                  metrics='mae',)"""

# Loading the best model after training
cnn_model = keras.models.load_model('Linear_Models/cnn_model.h5')

hist_cnn = cnn_model.fit(x_train,
                         y_label,
                         epochs=500,
                         validation_data=(val_data, val_label),
                         callbacks=[reduce_lr, stop_early])

#cnn_model.save('cnn_model.h5')

plot_hist(cnn_model)

# Finding the mse, mae, rmse of the CNN model
cnn_metrics = metrics(val_label, cnn_model.predict(val_data))

cnn_pred = rescale_data(val_data, Scaler, cnn_model)

# Plot the prediction from the CNN model vs the validation label
plot_pred(rescale_data(val_label, Scaler), cnn_pred)

# Creating a Bi-Directional LSTM model
"""lstm = lstm_model()
lstm.compile(loss='mae',
             optimizer='adam',
             metrics='mae')
"""

# Load the best trained model
lstm = keras.models.load_model('Linear_Models/Bi_LSTM.h5')


hist_lstm = lstm.fit(x_train,
                     y_label,
                     epochs=300,
                     validation_data=(val_data, val_label),
                     callbacks=[reduce_lr, stop_early])

#lstm.save('Bi_LSTM.h5')

plot_hist(hist_lstm)

# Find the mae, mse, rmse of the LSTM model
lstm_metrics = metrics(val_label, lstm.predict(val_data))

lstm_pred = rescale_data(val_data, Scaler, lstm)

# Plot the prediction from LSTM model vs the validation label
plot_pred(rescale_data(val_label, Scaler), lstm_pred)

# Plot the prediction from all 4 models against validation label
plt.figure(figsize=(10, 7))
month_range = np.arange(245, 304)
plt.ion()
plt.plot(month_range, np.array(rescale_data(val_label, Scaler)))
plt.plot(naive_predict[245:304])
plt.plot(month_range, dense_pred)
plt.plot(month_range, cnn_pred)
plt.plot(month_range, lstm_pred)
plt.xlabel(f'{len(dense_pred)} Months from April 2022')
plt.ylabel('Predicted CPI')
plt.legend(['Validation Label', 'Naive Model', 'Dense Linear Model', 'CNN Model', 'Bi-Directional LSTM'])
plt.show()

# Plot the error from each models
naive_error = [naive_metric[error] for error in naive_metric]
dense_error = [dense_metrics[error] for error in dense_metrics]
cnn_error = [cnn_metrics[error] for error in cnn_metrics]
lstm_error = [lstm_metrics[error] for error in lstm_metrics]

plt.bar(np.arange(len(naive_error)), naive_error, label='Naive', width=0.2)
plt.bar(np.arange(len(naive_error)) + 0.2, dense_error, label='Dense Model', width=0.2)
plt.bar(np.arange(len(naive_error)) + 0.4, cnn_error, label='CNN Model', width=0.2)
plt.bar(np.arange(len(naive_error)) + 0.6, lstm_error, label='Bi-Directional LSTM', width=0.2)
plt.xticks([0.3, 1.3, 2.3], ['MAE', 'MSE', 'RMSE'])
plt.legend()
