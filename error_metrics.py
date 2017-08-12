import numpy as np

def MAE(y, yhat, throw=0):
    """
    Mean Absolute Error (MAE)

    Defintion: Measures the average magnitude of the error in a set of
    predictions, witout considering the direction.

    Args:
        y (list type): target values
        yhat (list type): predictions
        throw (int): first X values, u dont want to consider caclulate in error

    Returns:
        mae (float): mean absolute error
    """
    y = y[throw:]
    yhat = yhat[throw:]
    return np.mean( np.abs(y-yhat) )

def RMSE(y, yhat, throw=0):
    """
    Root Mean Squared Error (RMSE)

    Defintion: Quadratic scoring rule that also measures the average
    magnitude of the error.

    Args:
        y (list type): target values
        yhat (list type): predictions
        throw (int): first X values, u dont want to consider caclulate in error

    Returns:
        rmse (float): root mean squared error
    """
    y = y[throw:]
    yhat = yhat[throw:]
    return np.sqrt( np.mean( np.square(y-yhat) ) )

# TODO
def NRMSE(y, yhat, throw=0):
    """
    Normalized Root Mean Squared Error (NRMSE)

    Defintion:

    Args:
        y (list type): target values
        yhat (list type): predictions
        throw (int): first X values, u dont want to consider caclulate in error

    Returns:
        nrmse (float): normalized root mean squared error
    """
    y = y[throw:]
    yhat = yhat[throw:]
    return np.sqrt( np.divide( np.mean( np.square(y-yhat) ), np.var(y) ) )
