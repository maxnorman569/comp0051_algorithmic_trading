# numeric imports
import numpy as np

def buy_and_hold_strategy(
    price_series : np.ndarray,
    initial_cash : float = 1.,
    ) -> np.ndarray:
    """
    Performs a buy and hold strategy on a given price series.

    Arguments:
    ----------
    price_series    : {np.ndarray}
                        > An array of prices.
    initial_cash    : {float}
                        > The initial cash to start with.
    
    Returns:
    ----------
    return_series   : {np.ndarray}
                        > An array of returns.
    """
    strategy_returns_train = (price_series/price_series[0]) * initial_cash

    return strategy_returns_train

