# numeric imports
import torch
import numpy as np

# data imports
import pandas as pd

# plotting imports
import matplotlib.pyplot as plt

# utility imports
from utilities.processing_utility import get_moving_average

# misc imports
import datetime

# typing imports
from typing import List, Tuple, Dict


def moving_average_mean_reversion_strategy(
        normalised_daily_excess_returns : np.ndarray,
        price_series : np.ndarray,
        window_size : int = 10,
        boillinger_band_threshold : float = 1.5,
        initial_cash : float = 1.,
        ) -> Tuple[np.ndarray, np.ndarray]:
    """ 
    Performs a moving average mean reversion strategy on a given 
    price series and corresponding daily excess returns.

    Arguments:
    ----------
    daily_excess_returns    : {np.ndarray}
                                > An array of normalised daily excess returns.
    price_series            : {np.ndarray}
                                > An array of prices.
    window_size             : {int}
                                > The window size for the moving average.
    boillinger_band_threshold : {float}
                                > The threshold for the boillinger band.
    initial_cash            : {float}
                                > The initial cash to start with.

    Returns:
    --------
    cash_series             : {np.ndarray}
                                > An array of cash values.
    """

    assert len(normalised_daily_excess_returns) == len(price_series), "The length of the daily excess returns and price series must be the same."

    # get moving average and boillinger bands
    ma_bb_data = get_moving_average(
                    series = pd.Series(normalised_daily_excess_returns),
                    ma_window = window_size, 
                    bollinger_bands = True, 
                    threshold = boillinger_band_threshold)

    # convert to numpy arrays
    moving_average = ma_bb_data['ma'].to_numpy()
    upper_boillinger_band = ma_bb_data['upper_bb'].to_numpy()
    lower_boillinger_band = ma_bb_data['lower_bb'].to_numpy()

    # initialise No. of shares & cash
    w = np.zeros(np.shape(price_series))
    cash = np.zeros(np.shape(price_series))
    cash[0] = initial_cash

    # backtest strategy
    for i, r in enumerate(normalised_daily_excess_returns[:-1]):

        # get current values
        ma = moving_average[i]
        upper_bb = upper_boillinger_band[i]
        lower_bb = lower_boillinger_band[i]

        current_price = price_series[i]

        # if we have insufficient data -> do nothing
        if np.isnan(ma):
            w[i+1] = w[i]
            cash[i+1] = cash[i]
            continue

        # if we are inside the bollinger band -> hold
        if (r >= lower_bb) and (r <= upper_bb):
            w[i+1] = w[i]
            cash[i+1] = cash[i]
            continue
        
        # if we are below the bollinger band -> buy
        if r < lower_bb:
            w[i+1] = cash[i] / current_price + w[i]
            cash[i+1] = 0
            continue
        
        # if we are above the bollinger band - sell
        if r > upper_bb:
            cash[i+1] = w[i] * current_price + cash[i]
            w[i+1] = 0
            continue
    
    # strategy returns
    strategy_returns = (price_series * w) + cash

    return strategy_returns, w, cash


def plot_strategy(
    plot_axis : plt.Axes,
    normalised_daily_excess_returns : np.ndarray,
    window_size : int = 10,
    boillinger_band_threshold : float = 1.5,
    ) -> None:
    """
    Plots the moving average mean reversion strategy.
    (i.e plots buy and sell calls with bollinger bands)

    Arguments:
    ----------
    plot_axis                   : {plt.Axes}
                                    > The axis to plot on.
    daily_excess_returns        : {np.ndarray}
                                    > An array of normalised daily excess returns.
    window_size                 : {int}
                                    > The window size for the moving average.
    boillinger_band_threshold   : {float}
                                    > The threshold for the boillinger band.

    Returns:
    ----------
    None
    """

    # get moving average and boillinger bands
    ma_bb_data = get_moving_average(
                    series = pd.Series(normalised_daily_excess_returns),
                    ma_window = window_size, 
                    bollinger_bands = True, 
                    threshold = boillinger_band_threshold)
    
    # convert to numpy arrays
    moving_average = ma_bb_data['ma'].to_numpy()
    upper_boillinger_band = ma_bb_data['upper_bb'].to_numpy()
    lower_boillinger_band = ma_bb_data['lower_bb'].to_numpy()

    # get masks
    sell_mask = (normalised_daily_excess_returns > upper_boillinger_band)
    buy_mask = (normalised_daily_excess_returns < lower_boillinger_band)

    # get plot range
    _range = np.arange(len(normalised_daily_excess_returns))

    # plot daily excess returns
    plot_axis.plot(normalised_daily_excess_returns, color = 'black', lw = 0.8)

    # plot moving average
    plot_axis.plot(moving_average, color = 'steelblue', lw = 0.8)

    # plot boillinger bands
    plot_axis.fill_between(_range, upper_boillinger_band, lower_boillinger_band, color = 'steelblue', alpha = 0.1)

    # plot buy calls
    plot_axis.scatter(_range[buy_mask], normalised_daily_excess_returns[buy_mask], marker = 'x', color = 'green')

    # plot sell calls
    plot_axis.scatter(_range[sell_mask], normalised_daily_excess_returns[sell_mask], marker = 'x', color = 'red')


def cross_validation(
        normalised_daily_excess_returns : np.ndarray,
        price_series : np.ndarray,
        window_size_sweep : np.ndarray,
        boillinger_band_threshold_sweep : np.ndarray,
        initial_cash : float = 1.,
        ) -> float:
    """
    Performs a k-fold cross validation on a mean reversion strategy 
    to find optimal window size and boillinger band threshold.

    Arguments:
    ----------
    normalised_daily_excess_returns    : {np.ndarray}
                                            > An array of normalised daily excess returns.
    price_series                        : {np.ndarray}
                                            > An array of prices.
    k                                   : {int}
                                            > The number of folds.
    window_size                         : {int}
                                            > The window size for the moving average.
    boillinger_band_threshold           : {float}
                                            > The threshold for the boillinger bands.
    initial_cash                        : {float}
                                            > The initial cash to start with.

    Returns:
    --------
    optim_window                        : {float}
                                            > The optimal window size.
    optim_threshold                     : {float}
                                            > The optimal boillinger band threshold.
    """

    # define the eval function
    eval_func = lambda x : x / initial_cash

    # create meshgrid
    window_i, bb_j = np.meshgrid(window_size_sweep, boillinger_band_threshold_sweep, indexing = 'ij')

    # create eval array
    eval_vec = np.zeros(np.shape(np.ravel(window_i)))

    # perform parameter sweep
    for idx, (window, threshold) in enumerate(zip(np.ravel(window_i), np.ravel(bb_j))):

        strategy_returns, _, _ = moving_average_mean_reversion_strategy(
            normalised_daily_excess_returns = normalised_daily_excess_returns,
            price_series = price_series,
            window_size = np.ravel(window_i)[idx],
            boillinger_band_threshold = np.ravel(bb_j)[idx],
            initial_cash = initial_cash)
        
        eval_vec[idx] += eval_func(strategy_returns[-1])

        print(f"Window Size: {window:^15} | Boillinger Band Threshold: {threshold:^15.3f} | Returns: {eval_vec[idx]:^15}")


    # get optimal parameters
    optim_window = np.ravel(window_i)[np.argmax(eval_vec)]
    optim_threshold = np.ravel(bb_j)[np.argmax(eval_vec)]

    return optim_window, optim_threshold