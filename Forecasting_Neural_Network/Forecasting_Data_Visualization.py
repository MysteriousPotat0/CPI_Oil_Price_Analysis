# Importing libraries
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np


# Function to calculate the auto-correlation function
def acf_plot(data_frame):
    """
    Takes in dataframe and plot the auto-correlation using statsmodel
    """
    plt.ion()
    # Plot the ACF
    sm.graphics.tsa.plot_acf(data_frame, lags=300, alpha=0.05)


# Plots the history object from keras model
def plot_hist(hist):
    """
    Takes in the keras history object and plots the loss, val_loss
    """

    plt.ion()
    plt.figure(figsize=(10, 7))
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss After Each Epoch')
    plt.legend(['Train', 'Validation'])
    plt.draw()


# Plot prediction vs ground truth
def plot_pred(true_value, predicted_value):
    """
    Plots true value against the predicted value
    """

    plt.ion()
    plt.figure(figsize=(10, 7))
    plt.plot(true_value)
    plt.plot(predicted_value)
    plt.ylabel('CPI Value')
    plt.title('Plot of predicted value against the true value')
    plt.legend(['True Value', 'Predicted Value'])
    plt.draw()
