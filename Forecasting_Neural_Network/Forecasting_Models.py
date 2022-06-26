import tensorflow as tf
from keras import layers
import keras.metrics


# Naives forecast
def naives(input):
    """
    Takes in the input and makes prediction based on previous observation
    """
    return input.shift(1)


# Fully connect dense model with linear output
def dense_linear():
    input = keras.Input(shape=(12, 32))
    x = keras.layers.Dense(256, activation='relu')(input)
    x = keras.layers.GlobalMaxPool1D()(x)
    x = layers.Dropout(0.5)(x)
    output = keras.layers.Dense(2)(x)

    return keras.Model(input, output, name='Dense_Model')


# CNN model with sliding window
def CNN_model():

    input = layers.Input(shape=(12, 32))
    x = layers.Conv1D(filters=8, kernel_size=6, padding='causal', activation='relu')(input)
    x = layers.GlobalAvgPool1D()(x)
    x = keras.Model(input, x)

    x2 = layers.Conv1D(filters=8, kernel_size=3, padding='causal', activation='relu')(input)
    x2 = layers.GlobalAvgPool1D()(x2)
    x2 = keras.Model(input, x2)

    x3 = layers.Concatenate(name='concat')([x.output, x2.output])

    x3 = layers.Dense(4, activation='relu')(x3)

    output = layers.Dense(1)(x3)

    return keras.Model(input, output, name='CNN_Model')


# LSTM model
def lstm_model():

    input = layers.Input(shape=(12, 32))
    x = layers.Bidirectional(layers.LSTM(units=8, activation='relu', return_sequences=True))(input)
    x = layers.GlobalAvgPool1D()(x)
    x = layers.Dense(8, activation='relu')(x)
    x = keras.Model(input, x)

    x2 = layers.Bidirectional(layers.LSTM(units=4, activation='relu', return_sequences=True))(input)
    x2 = layers.GlobalAvgPool1D()(x2)
    x2 = layers.Dense(2, activation='relu')(x2)
    x2 = keras.Model(input, x2)

    combined = layers.Concatenate(name='Concat')([x.output, x2.output])

    x3 = layers.Bidirectional(layers.LSTM(units=2, activation='relu', return_sequences=True))(input)
    x3 = layers.GlobalAvgPool1D()(x3)
    x3 = keras.Model(input, x3)

    combined = layers.Concatenate(name='Concat2')([combined, x3.output])

    output = layers.Dense(1)(combined)

    return keras.Model(input, output)
