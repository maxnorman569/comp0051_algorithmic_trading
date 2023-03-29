def evaluate_strategy(df, initial_capital=200_000, L=5):
    """
    Evaluate a trading strategy. Inputs a dataframe with columns SPDR, EFFR, 
    excess return and signal. Outputs a dataframe with columns V, cash_start, cash_end, theta, V_asset, leverage, 
    excess_return and excess_return_leveraged.

    # sell (everything) => signal = -1
    # buy (everything) => signal = 1
    """

    # Initialise
    data = df.copy()

    data['V'] = initial_capital 

    data['cash_start'] = initial_capital
    data['cash_end'] = initial_capital
    data['theta'] = 0
    data['V_asset'] = 0
    data['leverage'] = 0

    # Short adjusted return (using %s a bit funny with shorts)
    data['excess_return_adjusted'] = data['excess_return'] * np.sign(data['signal'])

    # Check if the signal is valid
    assert np.all(np.abs(data['signal']) <= 1), 'Signal must be between -1 and 1'

    # Loop through the data
    for t in range(1, len(data)):
        
        # Calculate the value to go long/short
        data.loc[data.index[t], 'theta'] = data.loc[data.index[t-1], 'V'] * L * data.loc[data.index[t], 'signal']

        # Calculate the amount of leverage used
        data.loc[data.index[t], 'leverage'] = np.abs(data.loc[data.index[t], 'theta'] * ((L-1)/L))

        # Calculate the value of unused cash
        data.loc[data.index[t], 'cash_start'] = data.loc[data.index[t-1], 'V'] * (1-np.abs(data.loc[data.index[t], 'signal']))

        # Returns on cash
        data.loc[data.index[t], 'cash_end'] = data.loc[data.index[t], 'cash_start'] * (1 + data.loc[data.index[t], 'EFFR'])

        # Unlevered returns (V_t)
        data.loc[data.index[t], 'V_asset'] =  np.abs(data.loc[data.index[t], 'theta'] * (1 + data.loc[data.index[t], 'excess_return_adjusted'])) - data.loc[data.index[t], 'leverage']

        # Calculate V (unlevered book value) (V^{total}_t)
        data.loc[data.index[t], 'V'] = data.loc[data.index[t], 'cash_end'] + data.loc[data.index[t], 'V_asset']
    
    # Calculate daily total PnL
    data['daily_PnL'] = data['V'] - data['V'].shift(1)
    data['daily_PnL'] = data['daily_PnL'].fillna(0)

    # Calculate cumulative PnL
    data['cumulative_PnL'] = data['daily_PnL'].cumsum()

    # Calculate accumulated values of change in cash, asset and total
    data['cumulative_cash_change'] = (data['cash_end'] - data['cash_start']).cumsum()
    data['cumulative_total_change'] = (data['V'] - data['V'].shift(1)).fillna(0).cumsum()
    data['cumulative_asset_change'] = data['cumulative_total_change'] - data['cumulative_cash_change']

    return data