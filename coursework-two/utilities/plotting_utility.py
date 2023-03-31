# numerical imports
import numpy as np

# data imports
import pandas as pd

# plotting impots
import matplotlib.pyplot as plt
import matplotlib.figure as figure
from PIL import Image

# misc imports
import os
import shutil

# typing imports
from typing import List


def plot_daily(
        df : pd.DataFrame,
    ) -> figure.Figure:
    """ 
    Plots the daily returns, risk free rate and excess returns.
    """

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex=True, figsize=(20, 10))

    # plot the SPDR returns
    ax1.set_title('Daily SPDR Returns')
    ax1.plot(df.index, df['daily_returns'], color='black', lw=.8, alpha=1, label='Daily SPDR Returns')

    # plot the risk free rate
    ax2.set_title('Daily Risk Free Rate')
    ax2.plot(df.index, df['daily_effr'], color='black', lw=1, alpha=.8, label='Daily Risk Free Rate')

    # plot the excess returns
    ax3.set_title('Daily Excess Returns')
    ax3.plot(df.index, df['daily_excess_returns'], color='black', lw=.8, alpha=1, label='Daily Excess Returns')

    # CUMMULATIVE RETURNS

    # plot the SPDR returns
    ax4.set_title('Cummulative SPDR Returns')
    ax4.plot(df.index, np.cumsum(df['daily_returns'].values), color='black', lw=.8, alpha=1, label='Cummulative SPDR Returns')

    # plot the risk free rate
    ax5.set_title('Cummulative Risk Free Rate')
    ax5.plot(df.index, np.cumsum(df['daily_effr'].values), color='black', lw=.8, alpha=1, label='Cummulative Risk Free Rate')

    # plot the excess returns
    ax6.set_title('Cummulative Excess Returns')
    ax6.plot(df.index, np.cumsum(df['daily_excess_returns'].values), color='black', lw=.8, alpha=1, label='Cummulative Excess Returns')

    plt.suptitle('Daily and Cummulative Return series', fontsize=16, fontweight='bold')
    fig.text(0.5, 0.02, r'Time $t$', ha='center', fontsize=14)
    fig.text(0.09, 0.5, 'Rate (%)', va='center', rotation='vertical', fontsize=14)

    return fig


def plot_price_and_returns(
    X_train : np.ndarray,
    P_train : np.ndarray,
    R_train : np.ndarray,
    X_test : np.ndarray,
    P_test : np.ndarray,
    R_test : np.ndarray,
    ) -> figure.Figure:
    """ 
    Plots the price and daily excess returns series.

    Arguments:
    ----------
    X_train     : {np.ndarray}
                    > The training inputs.
    P_train     : {np.ndarray}
                    > The training price series.
    R_train     : {np.ndarray}
                    > The training daily excess returns series.
    X_test      : {np.ndarray}
                    > The test inputs.
    P_test      : {np.ndarray} 
                    > The test price series.
    R_test      : {np.ndarray} 
                    > The test daily excess returns series.
    
    Returns:
    ----------
    fig         : {figure.Figure}
                    > The figure containing the plots.
    
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(20, 7))

    # plot the original price series
    ax1.axvline(x=len(X_train), color='red', lw=1, alpha=1, label='Training-Testing Split')
    ax1.plot(X_train, P_train, color='black', lw=1.5, alpha=1, )
    ax1.plot(X_test, P_test, color='black', lw=1.5, alpha=1, )

    ax1.set_title('S&P 500 Adjusted Closing Price', fontsize=16, fontweight='bold')
    # ax1.set_xlabel('Time', fontsize=14)
    ax1.set_ylabel('Price (USD)', fontsize=14)
    ax1.grid(True)
    ax1.legend(labels = ['Training-Testing Split'], loc = 'upper left')

    # plot the normalised excess returns
    ax2.axvline(x = len(X_train), color = 'red', lw = 0.8, alpha = 1, label='Training-Testing Split')
    ax2.plot(X_train, R_train, color = 'black', lw = 0.8, alpha = 1, )
    ax2.plot(X_test, R_test, color = 'black', lw = 0.8, alpha = 1, )
    ax2.set_title('S&P 500 Daily Excess Returns', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Time', fontsize=14)
    ax2.set_ylabel('Daily Excess Returns (%)', fontsize=14)
    ax2.grid(True)
    ax2.legend(labels = ['Training-Testing Split'], loc = 'upper left')

    return fig


def plot_strategy_positions(
    strategy_name : str,
    x_train : np.ndarray,
    train_theta : np.ndarray,
    train_V_asset : np.ndarray,
    x_test : np.ndarray,
    test_theta : np.ndarray,
    test_V_asset : np.ndarray,
    ) -> figure.Figure:
    """
    Plots the strategy positions on the training and test sets.

    Arguments:
    ----------
    strategy_name   : {str}
                        > Name of the strategy.
    x_train         : {np.ndarray}
                        > The training inputs.
    train_theta     : {np.ndarray}
                        > The training strategy positions.
    train_V_asset   : {np.ndarray}
                        > The training strategy asset value.
    x_test          : {np.ndarray}
                        > The test inputs.
    test_theta      : {np.ndarray}
                        > The test strategy positions.
    test_V_asset    : {np.ndarray}
                        > The test strategy asset value.
        
    Returns:
    ----------
    fig             : {figure.Figure}
                        > The figure containing the plots.
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # plot position on training set
    ax1.plot(x_train, train_theta, color = 'black', lw = 1, alpha = 1, label = r'$\theta_t$')
    ax1.fill_between(x_train, -5*train_V_asset, 5*train_V_asset, color = 'red', alpha = 0.2, label = r'$[-V \cdot L, V \cdot L]$')
    ax1.set_title('Training Set')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Position in dollars (USD)')
    ax1.legend()
    ax1.grid(True)

    # plot position on test set
    ax2.plot(x_test, test_theta, color = 'blue', lw = 1, alpha = 1, label = r'$\theta_t$')
    ax2.fill_between(x_test, -5*test_V_asset, 5*test_V_asset, color = 'red', alpha = 0.2, label = r'$[-V \cdot L, V \cdot L]$') 
    ax2.set_title('Test Set')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Position in dollars (USD)')
    ax2.legend()
    ax2.grid(True)

    plt.suptitle(r'{} long / shot positions $\theta_t$'.format(strategy_name), fontsize=16, fontweight='bold')

    return fig


def plot_pnl(
    strategy_name : str,   
    training_set : bool, 
    x : np.ndarray,
    delta_V_asset : np.ndarray,
    delta_V_cap : np.ndarray,
    delta_V_total : np.ndarray,
    ) -> figure.Figure:
    """ 
    Plots the strategy PnL on the training set.

    Arguments:
    ----------
    strategy_name       : {str}
                            > Name of the strategy.
    training_set        : {bool}
                            > Whether the PnL is on the training set.
    x_train             : {np.ndarray}
                            > The training inputs.
    train_delta_V_asset : {np.ndarray}
                            > The training strategy asset PnL.
    train_delta_V_cap   : {np.ndarray}
                            > The training strategy capital account PnL.
    train_delta_V_total : {np.ndarray}
                            > The training strategy total PnL.
    
    Returns:
    ----------
    fig                 : {figure.Figure}      
                            > The figure containing the plots.
    """
    set_name = 'Training' if training_set else 'Testing'
    color = 'black' if training_set else 'blue'

    # training holdings
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, sharex=True, sharey = False, figsize = (15, 7))

    ax1.plot(x, delta_V_asset, color = color, lw = 1, alpha = 1)
    ax1.set_title('Daily Trading PnL')
    ax1.grid(True)

    ax2.plot(x, delta_V_cap, color = color, lw = 1, alpha = 1)
    ax2.set_title('Daily Money Market Capital Account PnL')
    ax2.grid(True)

    ax3.plot(x, delta_V_total, color = color, lw = 1, alpha = 1)
    ax3.set_title('Daily Total Holdings PnL')
    ax3.grid(True)

    # CUMMULATIVE

    ax4.plot(x, np.cumsum(delta_V_asset), color = color, lw = 1, alpha = 1)
    ax4.set_title('Cumulative Trading PnL')
    ax4.grid(True)

    ax5.plot(x, np.cumsum(delta_V_cap), color = color, lw = 1, alpha = 1)
    ax5.set_title('Cumulative Money Market Capital Account PnL')
    ax5.grid(True)


    ax6.plot(x, np.cumsum(delta_V_total), color = color, lw = 1, alpha = 1)
    ax6.set_title('Cumulative Total Holdings PnL')
    ax6.grid(True)

    # sup labels
    set_name = 'Training' if training_set else 'Testing'
    plt.suptitle('Daily and Cumulative PnL for {} Strategy in {} Period'.format(strategy_name, set_name), fontsize = 16, fontweight = 'bold')
    fig.text(0.5, 0.03, r'Time $t$', ha='center', fontsize=14)
    fig.text(0.065, 0.5, 'PnL (USD)', va='center', rotation='vertical', fontsize=14)

    return fig


def save_to_gif(
    figurelist : List[plt.Figure],
    gifname : str,
    gifduration : int = 500,
    ) -> None:
    """
    Saves a gif from a given figure list.

    Arugments:
    ----------
    figure_list     : {List[plt.Figure]}
                        > List of figures to save as a gif.
    gif_name        : {str}
                        > Name of the gif to be saved.
    gif_duration    : {int}
                        > Duration of each frame in the gif.
    
    Returns:
    ----------
    None
    """
    # check if temp directory exists
    if os.path.exists("temp"):
        # delete the directory
        shutil.rmtree("temp")
        os.mkdir("temp")
    else: 
        os.mkdir("temp")

    # Create a list of image file names
    image_filenames = []
    for i, fig in enumerate(figurelist):
        filename = f"figure_{i}.png"
        fig.savefig(os.path.join("temp", filename))
        image_filenames.append(filename)

    # Open the first image and get its size
    with Image.open(os.path.join("temp", image_filenames[0])) as im:
        width, height = im.size

    # Create a new image object with the same size as the first image
    gif_image = Image.new('RGB', (width, height))

    # Open each image file and add it to the GIF image
    gif_frames = []
    for filename in image_filenames:
        with Image.open(os.path.join("temp", filename)) as im:
            gif_frames.append(im.copy())
    
    # Save the GIF
    gif_image.save(gifname, save_all=True, append_images=gif_frames, duration=gifduration, loop=50)

    # Delete the temporary directory
    shutil.rmtree("temp")