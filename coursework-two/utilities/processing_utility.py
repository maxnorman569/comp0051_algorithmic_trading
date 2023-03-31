# numeric imports
import torch
import numpy as np

# data import
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf

# misc imports
import datetime

# typing imports
from typing import List, Tuple, Dict


########## GET DATA FUNCTIONS ##########

def get_closing_price_from_yfinance( 
        ticker_symbol : str, 
        start_date : datetime.datetime, 
        end_date : datetime.datetime,
        adjusted : bool = True,
        ) -> pd.DataFrame:
    """
    Returns a pandas dataframe of the closing prices for a given stock ticker

    Arguments:
    ---------- 
    ticker_symbol   : {string}
                        > The stock ticker symbol 
    start_date      : {datetime.datetime}
                        > The start date of the data
    end_date        : {datetime.datetime}  
                        > The end date of the data
    adusted         : {bool}
                        > Whether to return adjusted closing prices or not
    
    Returns:
    ----------
    df              : {pandas.DataFrame}
                        > A pandas dataframe of the closing prices for the given stock ticker
    """
    # get data
    df = yf.download(ticker_symbol, start_date, end_date)

    # adjust index
    df.index = df.index.date

    # return specified data
    if adjusted:
        adj_close_p_df = pd.DataFrame(df['Adj Close'])
        adj_close_p_df.columns = ['adjusted_close_price']
        return adj_close_p_df
    else:
        close_p_df = pd.DataFrame(df['Close'])
        close_p_df.columns = ['close_price']
        return close_p_df


def get_effr_from_nyfed(
        start_date : datetime.datetime, 
        end_date : datetime.datetime,
        daily : bool = False,
        ) -> pd.DataFrame:
    """
    Returns a pandas dataframe of the effective federal funds rate 

    Arguments:
    ---------- 
    start_date      : {datetime.datetime}
                        > The start date of the data
    end_date        : {datetime.datetime}  
                        > The end date of the data
    daily           : {bool}
                        > Whether to return daily or annualised data
    
    Returns:
    ----------
    df              : {pandas.DataFrame}
                        > A pandas dataframe of the effective federal funds rate
    """
    # get data
    df = web.DataReader('EFFR', 'fred', start_date, end_date)

    # adjust index
    df.index = df.index.date

    if daily == True:
        # convert to % and ajust for 252 trading days
        df.columns = ['daily_effr']
        return df / (100*252)
    else:
        # convert to %
        df.columns = ['annual_effr']
        return df / 100
    

def get_daily_excess_return(
        closing_price : pd.DataFrame,
        daily_risk_free_rate : pd.DataFrame
    ) -> np.ndarray:
    """
    Returns a numpy array of the daily excess returns for a given stock and daily risk-free rate

    Arguments:
    ---------- 
    closing_price           : {pandas.DataFrame indexed by date}
                                > A pandas dataframe of the closing prices for a given stock
    daily_risk_free_rate    : {pandas.DataFrame indexed by date}
                                > A pandas dataframe of the daily risk-free rate
    
    Returns:
    ----------
    df                      : {pandas.DataFrame indexed by date}
                                > A pandas dataframe for the daily excess returns for a given stock
    """
    
    # get prcing column
    price_column = closing_price.columns[0]
    daily_effr_column = daily_risk_free_rate.columns[0]

    # concatenate data
    data = closing_price.join(daily_risk_free_rate, how = 'left')
    # data = pd.concat([closing_price, daily_risk_free_rate], axis=1)

    # drop na
    # data[daily_effr_column] = data[daily_effr_column].fillna(method = 'ffill').to_numpy()
    data[daily_effr_column].fillna(method='ffill', inplace=True)

    # get daily % change
    data['daily_returns'] = data[price_column].pct_change().fillna(0).to_numpy()

    # get excess returns
    data['daily_excess_returns'] = data['daily_returns'].values - data[daily_effr_column].values

    return data.fillna(0)


def normalise_data(
        data : np.ndarray
        ) -> np.ndarray:
    """
    Normalise data to zero mean and unit variance.
    
    Arguments:
    ----------
    data        : {np.ndarray}
                    > Data to be normalised.
    
    Returns:
    --------
    norm_data   : {np.ndarray}
                    > Normalised data.
    """
    norm_data = (data - np.mean(data)) / np.std(data)

    return norm_data


def get_q_2_cw_data():
    """
    Returns a pandas dataframe of all the data needed for question 2 of the coursework 

    Arguments:
    ---------- 
    None
    
    Returns:
    ----------
    df  : {pandas.DataFrame}
            > A pandas dataframe containing all data for quetion 2 of the coursework.
    """
    # get price data
    p_df = get_closing_price_from_yfinance(
                ticker_symbol = 'SPY',
                start_date = datetime.datetime(2014, 1, 1),
                end_date = datetime.datetime(2019, 12, 31),
                adjusted=True
                )

    # get risk-free rate data
    daily_effr_df = get_effr_from_nyfed(
                        start_date = datetime.datetime(2014, 1, 1),
                        end_date = datetime.datetime(2019, 12, 31),
                        daily=True)
    
    # get excess returns
    df = get_daily_excess_return(p_df, daily_effr_df)

    # get nomarlised excess returns
    df['normalised_excess_returns'] = normalise_data(df['daily_excess_returns'].to_numpy())

    # get compounded risk-free rate
    df['risk_free_rate_factor'] = df['daily_effr'].to_numpy() + 1.

    return df


########## GENERAL PROCESSING FUNCTIONS ##########

def get_train_test_split(
        splitindex : int,
        **series : torch.Tensor
        ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Returns a train and test split of the data.
    
    Arguments:
    ----------
    splitindex  : {int}
                    > The index to split the data at.
    **series    : {torch.Tensor}
                    > The data to be split.
    
    Returns:
    ----------
    train       : {torch.Tensor}
                    > The training data.
    test        : {torch.Tensor}
                    > The test data.
    """
    # get the key and values
    series_keys = series.keys()
    series_values = series.values()

    # get the train and test split
    train_dict = {key : values[:splitindex] for key, values in zip(series_keys, series_values)}
    test_dict = {key : values[splitindex:] for key, values in zip(series_keys, series_values)}

    return train_dict, test_dict


def get_moving_average(
    series : pd.Series,
    ma_window : int,
    bollinger_bands : bool = True,
    threshold : float = 2
    ) -> pd.DataFrame:
    """
    Returns a pandas dataframe of the moving average for a given window.
    If bollinger_bands is True, then the upper and lower bollinger bands are also returned.

    Arguments:
    ----------
    series      : {pandas.Seres}
                    > A pandas series of the data.
    ma_window   : {int}
                    > The moving average window.
    
    Returns:
    ----------
    df          : {pandas.DataFrame}
                    > A pandas dataframe of the moving average for a given window.
    """
    # get moving average
    ma_series = series.rolling(window=ma_window).mean()

    if bollinger_bands:
        # get upper and lower bollinger bands
        upper_bb = ma_series + (series.rolling(window=ma_window).std() * threshold)
        lower_bb = ma_series - (series.rolling(window=ma_window).std() * threshold)

        # concatenate data
        ma_series = pd.concat([ma_series, upper_bb, lower_bb], axis=1)
        ma_series.columns = ['ma', 'upper_bb', 'lower_bb']

    return ma_series


def get_rsi(P):
    delta = pd.DataFrame({'price' : P}).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean().fillna(0)
    avg_loss = loss.rolling(window=14).mean().fillna(0)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)


def get_macd(P):
    ema_12 = pd.DataFrame({'price' : P}).ewm(span=12, adjust=False).mean().fillna(0)
    ema_26 = pd.DataFrame({'price' : P}).ewm(span=26, adjust=False).mean().fillna(0)
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd.fillna(0)
