import datetime
import pandas as pd
import numpy as np

import pandas_datareader.data as web

import yfinance as yf


def get_closing_price_from_yfinance( 
        ticker_symbol : str, 
        start_date : datetime.datetime, 
        end_date : datetime.datetime
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
    
    Returns:
    ----------
    df              : {pandas.DataFrame}
                        > A pandas dataframe of the closing prices for the given stock ticker
    """
    # get data
    df = yf.download(ticker_symbol, start_date, end_date)

    # adjust index
    df.index = df.index.date
    
    
    return pd.DataFrame(df['Adj Close'])


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
    closing_price               : {pandas.DataFrame indexed by date}
                                        > A pandas dataframe of the closing prices for a given stock
    daily_risk_free_rate        : {pandas.DataFrame indexed by date}
                                        > A pandas dataframe of the daily risk-free rate
    
    Returns:
    ----------
    df                          : {pandas.DataFrame indexed by date}
                                        > A pandas dataframe for the daily excess returns for a given stock
    """
    # concatenate data
    data = pd.concat([closing_price, daily_risk_free_rate], axis=1)

    # drop na
    data = data.dropna()

    # get daily % change
    data['daily_returns'] = data['Adj Close'].pct_change()

    # get excess returns
    data['daily_excess_returns'] = data['daily_returns'] - data['daily_effr']

    return data.fillna(0)
