# Importing libraries
import numpy as np
from sklearn.model_selection import train_test_split
from Linear_Models import *
import statsmodels.api as sm
from Data_Visualization import *

# Import the csv files into pandas dataframe
CPI = pd.read_csv('Data/CPI_data_1997_to_2022.csv')
Energy_Prices = pd.read_csv('Data/Energy_Prices.csv')
Production_Data = pd.read_csv('Data/International_PetroleumProduction_Consumption_and_Inventories.csv')

# Take a look at the data structures
print(CPI[0: 10])
print(Energy_Prices[0: 10])
print(Production_Data[0: 10])

# Check how many columns that has the word prediction
print('Number of columns with word production:',
      len([production for production in Production_Data.columns if 'Production' in production]))

# Check how many columns that has the word consumption
print('Number of columns with word Consumption:',
      len([production for production in Production_Data.columns if 'Consumption' in production]))

# Plot the Production by production and consumption
plot_values(Production_Data.iloc[:, 0: 11])
plot_values(Production_Data.iloc[:, np.r_[0, 13:26]])

# Plot total world production vs total world consumption
plot_values(Production_Data[['DATE', 'Total_World_Production', 'Total_World_Consumption']])

# Plot the prices of different fuels
plot_values(Energy_Prices[['DATE', 'Gasoline_Price', 'Diesel_Fuel_Price', 'Fuel_Oil_Price']])

# Plot the fuel price vs brent price
plot_values(Energy_Prices[['DATE', 'Brent_Crude_Oil_Spot_Price', 'Fuel_Oil_Price']])

# Merge the cpi, energy prices, production data by date into one dataframe
All_data = CPI.merge(Energy_Prices, on='DATE').merge(Production_Data, on='DATE')
All_data = All_data.rename(columns={'VALUE': 'CPI_Value'})

# Plot the fuel prices vs the consumption
plot_values(All_data[['DATE', 'Total_World_Consumption', 'Gasoline_Price']])

# Plot CPI values over last 15 years
plot_values(All_data[['DATE', 'CPI_Value']])

# Calculating the inflation rate month over month
All_data['Inflation'] = ((All_data['CPI_Value'] - All_data.CPI_Value.shift(1)) / All_data.CPI_Value.shift(1)) * 100
All_data.loc[0, 'Inflation'] = 0

# Plot inflation
plot_values(All_data[['DATE', 'Inflation']])

# Plot the inflation values from month 290+
plot_values(All_data.loc[290:, ['DATE', 'Inflation']])

# Convert the date from YYYY-MM-DD into months starting at 1
All_data['DATE'] = range(1, All_data.shape[0] + 1)

# Plot a correlation heatmap
corr_heatmap(All_data)

# Separate the response and explanatory variables
x = All_data.loc[:, All_data.columns != 'CPI_Value']
y = All_data['CPI_Value']

# Use train test split from sklearn to chop off last 20% of the data for validation
train_x, val_x, train_y, val_y = train_test_split(x, y,
                                                  test_size=0.2,
                                                  shuffle=False)

# Import linear regression function from Linear_Regression_Model.py and fit the model
lm = linear_regression_model(train_x, train_y)  # Fit model using x and y

# Summary of the linear regression model
lm.summary()

# Linear regression analysis
sm.graphics.influence_plot(lm, criterion='cooks')

# From the influence plot, find the observation that has a large influence
influence = [0, 1, 36, 53, 54, 58, 60, 67, 71, 108, 138, 140, 142, 239]
influential = All_data.iloc[influence, :]

# normal plots for the linear regression
sm.qqplot(lm.resid, line='45', fit=True)

# Use the linear regression model to make prediction on the validation data
prediction = lm.predict(sm.add_constant(val_x))

# 95% confidence interval to dataframe
conf_int = pd.DataFrame(lm.conf_int(alpha=0.05).T)

train_val_plot(train_x, train_y, val_x, val_y, prediction)

print(lm.params.to_dict(), lm.pvalues >= 5e-2)

aggregated_values = aggregate_stats(All_data.iloc[:, 1:])

influential.iloc[:, 1:] - aggregated_values.loc['median', :]
