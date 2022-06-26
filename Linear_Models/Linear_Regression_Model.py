# Import the Linear regression from sklearn
import statsmodels.api as sm


# Create a linear regression model from a database input
def linear_regression_model(explanatory_data, response_data):
    """
    Takes in data from data frame and fit a linear regression model
    :param explanatory_data: All of the explanatory variables that is trying to explain the result
    :param response_data: The response data that is the result from the explanatory data
    :return: Linear regression model
    """

    # Creates the linear model and fit it to response and explanatory data
    x = sm.add_constant(explanatory_data)
    lm = sm.OLS(response_data, x).fit()

    return lm
