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


def get_ma_cross_over_signal(
    price_series: np.ndarray,
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
    

    # convert to numpy arrays
    buy_mask = torch.zeros_like(price_series)
    sell_mask = torch.zeros_like(price_series)

    buy_delta = torch.zeros_like(price_series)
    sell_delta = torch.zeros_like(price_series)

    signal = torch.zeros_like(price_series)   

    for i, p in enumerate(price_series):

        # if i < window_size:
        #     signal[i] = 0
        #     continue

        ma = price_series[i-window_size:i].sum() / window_size
        upper_bound = ma + (2 * price_series[i-window_size:i].std())
        lower_bound = ma - (2 * price_series[i-window_size:i].std())

        # clear position once we are within the bounds
        if ( p >= lower_bound)  and ( p <= upper_bound):
            
            if (signal[i-1] > 0) and (p > ma):
                signal[i] = signal[i-1]

            elif (signal[i-1] < 0) and (p < ma):
                signal[i] = signal[i-1]

            else:
                signal[i] = 0

        # buy
        elif p > lower_bound:
            buy_mask[i] = 1
            signal[i] = -1*(lower_bound - p)
            continue

        elif p < upper_bound:
            sell_mask[i] = 1
            signal[i] = -1*(upper_bound - p)
            continue

        else:
            signal[i] = 0

    # norm scale the signal
    scaled_signal = signal / torch.std(signal)
    scaled_signal = torch.clamp(scaled_signal, -2, 2)

    return scaled_signal / 2, buy_mask, sell_mask
        
