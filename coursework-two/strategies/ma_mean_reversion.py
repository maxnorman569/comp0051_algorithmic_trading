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


def get_ma_mean_reversion_signal(
    normalised_daily_excess_returns_series: np.ndarray,
    window_size : int, 
    ) -> np.ndarray:
    """
    Get the signal for a moving average mean reversion strategy.

    Arguments:
    ----------
    normalised_daily_excess_returns_series  : {np.ndarray}
                                                > The normalised daily excess returns series.

    window_size                             : {int}
                                                > The window size for the moving average.
    
    Returns:
    --------
    signal                                  : {np.ndarray}
                                                > The signal for the moving average mean reversion strategy.
    """
    # get moving average and boillinger bands
    ma_bb_data = get_moving_average(
                    series = pd.Series(normalised_daily_excess_returns_series),
                    ma_window = window_size, 
                    bollinger_bands = True, 
                    threshold = 2.)

    # convert to numpy arrays
    moving_average = ma_bb_data['ma'].to_numpy()
    upper = ma_bb_data['upper_bb'].to_numpy()
    lower = ma_bb_data['lower_bb'].to_numpy()

    # get signal
    signal = torch.zeros_like(normalised_daily_excess_returns_series)    

    buy_count = 0
    sell_count = 0

    for i, (value, ma, upper, lower) in enumerate(zip(normalised_daily_excess_returns_series, moving_average, upper, lower)):

        if i == 0:
            signal[i] = 0

        if (value < upper) and (value > lower):
            
            if (signal[i-1] > 0) and (value > ma):
                signal[i] = 0
            
            elif (signal[i-1] < 0) and (value < ma):
                signal[i] = 0
            
            else:
                signal[i] = signal[i-1]

        if value <= lower:
            buy_count += 1
            signal[i] = torch.clamp(value - lower, min = -1, max = None).item()

        if value >= upper:
            sell_count += 1
            signal[i] = torch.clamp(value - upper, min = None, max = 1).item()
    
    return np.array(signal)


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
    plot_axis.plot(moving_average, color = 'steelblue', lw = 1.)

    # plot boillinger bands
    plot_axis.fill_between(_range, upper_boillinger_band, lower_boillinger_band, color = 'steelblue', alpha = 0.1)

    # plot buy calls
    plot_axis.scatter(_range[buy_mask], normalised_daily_excess_returns[buy_mask], marker = 'x', color = 'green')

    # plot sell calls
    plot_axis.scatter(_range[sell_mask], normalised_daily_excess_returns[sell_mask], marker = 'x', color = 'red')
