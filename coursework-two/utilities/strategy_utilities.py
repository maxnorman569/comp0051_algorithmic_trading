# numeric imports
import numpy as np
import torch 

# typing imports
from typing import List, Tuple, Dict


def get_theta_and_V(
    daily_excess_return_series : np.ndarray,
    daily_risk_free_series : np.ndarray,
    signal : np.ndarray,
    initial_cash : float = 200_000.,
    leverage : float = 5.) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ 
    Evaluates a mean reversion strategy given a signal and returns the theta, V_asset, V_cap and V_total

    Arguments:
    ----------
    daily_excess_return_series  : {np.ndarray}
                                    > The daily excess return series
    daily_risk_free_series      : {np.ndarray}
                                    > The daily risk free rate series  
    signal                      : {np.ndarray}
                                    > The signal series
    initial_cash                : {float}
                                    > The initial cash
    leverage                    : {float}
                                    > The leverage

    Returns:
    ----------
    theta                       : {np.ndarray}
                                    > The dollar values (either long or short)
    V_asset                     : {np.ndarray}
                                    > The unlevereaged value of the assets.
    V_cap                       : {np.ndarray}
                                    > The value of the unused capital. 
    V_total                     : {np.ndarray}
                                    > The total value of the holdings.
    """
    # ge let n_days
    n_days = len(signal)

    # set up
    theta = np.zeros((n_days,))

    # assets
    V_t = np.zeros((n_days,))
    delta_V_t = np.zeros((n_days,))

    # money market
    V_cap = np.zeros((n_days,))
    delta_V_cap = np.zeros((n_days,))

    # total
    V_total = np.zeros((n_days,))
    delta_V_total = np.zeros((n_days,))

    # # initial values
    V_t[0] = initial_cash
    delta_V_t[0] = 0
    V_cap[0] = 0
    delta_V_cap[0] = 0
    V_total[0] = V_t[0] + delta_V_cap[0]
    delta_V_total[0] = 0

    # backtest strategy
    for i, s in enumerate(signal[:-1]):

        if i == 0:
            continue
        
       # get position size
        # theta[i] = V_t[i-1] * leverage * s
        theta[i] = V_total[i-1] * leverage * s

        # get V_t
        delta_V_t[i] = (daily_excess_return_series[i+1] * theta[i]) 
        V_t[i] = delta_V_t[i] + V_t[i-1]
        
        # get V_cap_t
        M = np.abs(theta[i]) / leverage
        delta_V_cap[i] = (V_total[i-1] - M) * (daily_risk_free_series[i+1]) 
        V_cap[i] = delta_V_cap[i] + V_cap[i-1]

        # Get V_total_t
        delta_V_total[i] = delta_V_t[i] + delta_V_cap[i]
        V_total[i] = delta_V_total[i] + V_total[i-1]

    strategy_data = {
        "theta" : theta,
        "V_t" : V_t,
        "delta_V_t" : delta_V_t,
        "V_cap" : V_cap,
        "delta_V_cap" : delta_V_cap,
        "V_total" : V_total,
        "delta_V_total" : delta_V_total}

    
    return strategy_data


def get_turnover_dollars(
    theta : np.ndarray,
    ) -> float:
    """ 
    Calculates the turnover in dollars traded over time

    Arguments:
    ----------
        theta       : {np.ndarray}
                        > The dollar values invested
    
    Returns:
    ----------
        turnover    : {float}
                       > The turnover in dollars traded over time
    """
    theta_delta = np.diff(theta)
    turnover = np.sum(np.abs(theta_delta))
    return turnover


def get_turnover_units(
    theta : np.ndarray, 
    price_series : np.ndarray
        ) -> float:
    """
    Calculates the turnover in units traded over time

    Arguments:
    ----------
        theta           : {np.ndarray}
                            > The dollar values invested
        price_series    : {np.ndarray}
                            > The price series
    
    Returns:
    ----------
        turnover        : {float}
                            > The turnover in units traded over time
    """
    theta_over_price = theta / price_series
    turnover = np.sum(np.abs(np.diff(theta_over_price)))
    return turnover


def get_daily_trading_pnl(
    unleveraged_asset_holdings : np.ndarray,
    ) -> np.ndarray:
    """ 
    Calculates the daily trading PnL for the strategy.

    Arguments:
    ----------
    unleveraged_asset_holdings  : {np.ndarray}
                                    > The daily holdings of the asset.

    Returns:
    --------
    daily_trading_pnl           : {np.ndarray}
                                    > The daily trading PnL.
    """
    # calculate the daily trading pnl
    daily_trading_pnl = np.diff(unleveraged_asset_holdings)
    return daily_trading_pnl


def get_daily_money_market_capital_account(
    money_market_capital_account : np.ndarray,
    ) -> np.ndarray:
    """
    Calculates the daily money market capital account for unused cash for the the strategy.
    
    Arguments:
    ----------
    money_market_capital_account        : {np.ndarray}
                                            > The daily money market capital account for unused cash.

    Returns:
    ----------
    daily_money_market_capital_account  : {np.ndarray}
                                            > The daily money market capital account for unused cash.
    """
    # calculate the daily money market capital account
    daily_money_market_capital_account = np.diff(money_market_capital_account)
    return daily_money_market_capital_account


def get_daily_total_holdings(
    total_holdings : np.ndarray,
    ) -> np.ndarray:
    """
    Calculates the daily total holdings for the strategy.

    Arguments:
    ----------
    total_holdings      : {np.ndarray}
                            > The daily total holdings for the strategy.
    
    Returns:
    ----------
    daily_total_holdings : {np.ndarray}
                            > The daily total holdings for the strategy.
    """
    # calculate the daily total holdings
    daily_total_holdings = np.diff(total_holdings)
    return daily_total_holdings