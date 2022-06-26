# Import matplot and datetime
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import pandas as pd


# A function that takes in a dataframe and plot the date against different values
def plot_values(input):
    """
    Takes in a dataframe and plot values against year and month
    """

    # Convert date into date time format in a list
    date = [datetime.datetime.strptime(time, '%Y-%m-%d') for time in input['DATE']]

    plt.figure()

    # Goes through each column of input
    for col_num, column in enumerate(input.columns[1:]):

        # Check if the loop reached the end of the dataframe
        if col_num == input.shape[1] - 1:
            break  # Break out of the loop

        # If loop did not hit the end of dataframe
        else:
            plt.plot(date, input[column], label=column)
            plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=6, fancybox=True)


# Function that plots the training data, validation data to check how well the model fits
def train_val_plot(train_x, train_y, val_x, val_y, prediction):
    """
    Takes in training data, validation data returns a plot of how well the model fits on validation data
    """

    plt.plot(train_x['DATE'], train_y)
    plt.scatter(val_x['DATE'], prediction)
    plt.scatter(val_x['DATE'], val_y)


# Correlation heat map
def corr_heatmap(input):
    """
    Takes in a dataframe and creates a correlation heat map
    :param input: Dataframe
    """
    plt.figure(figsize=(15, 9))
    plt.title('Correlation Heatmap')
    sns.heatmap(input.corr(), annot=True, cmap='RdYlGn', cbar=True, xticklabels=1, yticklabels=1, fmt='.1f')
    plt.tight_layout()
    plt.show()


# Function to find the summary statistic
def aggregate_stats(input):
    """
    Takes in a dataframe and returns the summary statistic in dataframe
    """

    # Statistic functions
    stats_fun = ['min', 'max', 'median', 'mean', 'quantile']

    # Aggregate statistic
    return pd.DataFrame(input.agg(stats_fun))
