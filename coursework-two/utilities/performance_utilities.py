import numpy as np


def sharpe_ratio(
        excess_returns : np.ndarray,
        ) -> float:
    """
    Calculates the sharpe ratio of a strategy

    Arguments:
    ----------
        excess_returns      : {np.ndarray}
                                > The excess returns of a strategy
                                (already discounted by the risk-free rate)
        
    Returns:
    ----------
        sharpe_ratio        : {float}
                                > The sharpe ratio of a strategy
    """
 
    # computes the sharpe ratio
    # sharpies = np.sum(excess_returns) / (np.std(excess_returns) * np.sqrt(len(excess_returns)))

    sharpies = ((np.mean(excess_returns)*len(excess_returns)) / (np.std(excess_returns)*np.sqrt(len(excess_returns))))
    
    return sharpies 


def sortino_ratio( 
        excess_returns : np.ndarray,
        ) -> float:
    """
    Calculates the sortino ratio of a strategy

    Arguments:
    ----------
        excess_returns      : {np.ndarray}
                                > The excess returns of a strategy
                                (already discounted by the risk-free rate)

    Returns:
    ----------
        sortino_ratio       : {float}
                                > The sortino ratio of a strategy
    """
    # computes the sortino ratio
    # sorties = np.sum(excess_returns) / (np.std(excess_returns[excess_returns < 0]) * np.sqrt(len(excess_returns)))

    sorties = ((np.mean(excess_returns)*len(excess_returns)) / (np.std(excess_returns[excess_returns < 0])*np.sqrt(len(excess_returns))))

    return sorties


def max_drawdown(
        daily_excess_returns : np.ndarray, 
        cummulative_returns : np.ndarray 
        ) -> float:
    """
    Calculates the maximum drawdown of a strategy as a percentage

    Arguments:
    ----------
        daily_excess_returns    : {np.ndarray}
                                    > The daily excess returns of a strategy
                                    (already discounted by the risk-free rate)
        cummulative_returns     : {np.ndarray}
                                    > The cummulative returns of a strategy

    Returns:
    ----------
        max_drawdown            : {float}
                                    > The maximum drawdown of a strategy
                                    (already discounted by the risk-free rate)
    """
    # Compute the percentage returns
    pect_returns = daily_excess_returns / cummulative_returns
    
    # Compute the cumulative returns and the running maximum
    cum_returns = np.cumprod(1 + pect_returns)
    running_max = np.maximum.accumulate(cum_returns)
    
    # Compute the drawdowns and find the maximum drawdown
    drawdowns = (cum_returns / running_max) - 1
    max_drawdown = np.min(drawdowns)

    return max_drawdown


def calmar_ratio( 
        daily_excess_returns : np.ndarray, 
        cummulative_returns : np.ndarray
        ) -> float:
    """
    Calculates the calmar ratio of a strategy

    Arguments:
    ----------
        daily_excess_returns    : {np.ndarray}
                                    > The daily excess returns of a strategy
                                    (already discounted by the risk-free rate)
        cummulative_returns     : {np.ndarray}
                                    > The cummulative returns of a strategy

    Returns:
    ----------
        calmar_ratio            : {float}
                                    > The calmar ratio of a strategy
    """
    # compute the maximum drawdown
    maximum_drawdown = max_drawdown(daily_excess_returns, cummulative_returns)

    # compute the percentage returns
    pct_returns = daily_excess_returns / cummulative_returns

    # compute the calmar ratio
    calmar = np.sum(pct_returns) / abs(maximum_drawdown)
    
    return calmar