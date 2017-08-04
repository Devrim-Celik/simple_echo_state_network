import numpy as np

def MAE(y, yhat):
    """
    Mean Absolute Error (MAE)

    Defintion: Measures the average magnitude of the error in a set of
    predictions, witout considering the direction.

    Args:
        y (list type): target values
        yhat (list type): predictions

    Returns:
        mae (float): mean absolute error
    """
    return np.mean( np.abs(y-yhat) )

def RMSE(y, yhat):
    """
    Root Mean Squared Error (RMSE)

    Defintion: Quadratic scoring rule that also measures the average
    magnitude of the error.

    Args:
        y (list type): target values
        yhat (list type): predictions

    Returns:
        rmse (float): root mean squared error
    """
    return np.sqrt( np.mean( np.square(y-yhat) ) )

# TODO
def NRMSE(y, yhat):
    """
    Normalized Root Mean Squared Error (NRMSE)

    Defintion:

    Args:
        y (list type): target values
        yhat (list type): predictions

    Returns:
        nrmse (float): normalized root mean squared error
    """
    return np.sqrt( np.divide( np.mean( np.square(y-yhat) ), np.var(y) ) )
