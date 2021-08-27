# import os
import re
import datetime
import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

def plot_forecast(y='y', yhat='yhat', 
                  ci_lower=None, ci_upper=None, 
                   
                  data=None, data_path=None, data_index_col=None, 
                  data_drop_duplicates=True, keep='last',
                   
                  start_date=None, end_date=None, 
                  date_format='%Y-%m-%d',
                   
                   plot_style='classic', 
                   plot_size=(20,12), plot_titlesize=18,
                   
                   y_color='k', y_ls='-', y_lw=1.5, 
                   y_marker='None', y_markersize=8, y_facecolor='k',
                   
                   yhat_color='#CC1D92', yhat_ls='--', yhat_lw=3,
                   yhat_marker='None', yhat_markersize=8, yhat_facecolor='#CC1D92',
                   
                   vline_color='k', vline_ls='-', vline_lw=2,
                   
                   ci_label='_nolegend_', ci_color='#FA71CD',
                   
                   print_mse=True):
    
    """
    A basic function to plot observed, predicted values. 
    ## Use is intended for team members. Client(s) should be interacting with interactive plots. ##
    ## Exception,errors are not handled well here ## 
    
    Parameters
    ----------
    y: str, default='y'
        Column name of observed values.
        
    yhat: str, default='yhat'
        Column name of predicted values.
        
    ci_lower: str, optional
        Column name of upper confidence interval.
        
    ci_upper: str, optional
        Column name of lower confidence interval.
    
    data: pd.DataFrame, optional
        DataFrame with dates and observed, predicted values.
        
    data_path: str, optional
        Path to load .csv file as pd.DataFrame.
        
    data_index_col: int, str, optional
        Column index or name to set as index.
        
    data_drop_duplicates: bool, default=True
        If True, drop duplicate index values.
    
    keep: str, default='last'
        Which entry to keep when dropping duplicates. # --> pd.DataFrame.drop_duplicates(keep='')
    
    start_date: str, optional
        Start date to subset data and plot (by default, expects '%Y-%m-%d').
        
    end_date: str, optional
        End date to subset data and plot (by default, expects '%Y-%m-%d').
        
    date_format: str, default='%Y-%m-%d'
        Format to convert start_date, end_date to datetime.
    
    plot_style: matplotlib style sheet, default='classic'
        Matplotlib plotting style to use.
        
    plot_size: tuple, default=(20,12)
        Size of figure output. Does not refer to size of individual plots.

    plot_titlesize: int, default=18
        Title size of plots. Also, affects label, tick size --> (label size = plot_titlesize - 6).
    
    print_mse: bool, default=True
        If True, print mse, rmse.
        
    ci_label: str, default='_nolegend_'
        Confidence interval label to show in legend.
        (e.g., ci_label='95%' shows '95% Confidence Interval' in legend)
        
    Line, marker style parameters
    -----------------------------
    ci_color: str, default='#FA71CD'
        Fill color.
    
    for x in [y, yhat, vline]:
    
        x_color: str
            Line color.
    
        x_marker: str, default='None'
            Type of marker.

        x_ls: str
            Linestyle.

        x_lw: int
            Linewidth.

        x_markersize: int
            Markersize.
            
        x_facecolor: str
            Marker face color.

    Return
    ------
    fig: matplotlib figure
    
    """
    
    # - - - - - - - - - - - - - - - Add - -  - - - - - - - - - - - - #
    # "'data' or 'data_path' is required."
    # if both provided, use 'data'
    # - - - - - - - - - - - - - - - Add - -  - - - - - - - - - - - - #
    
    # Try loading df from pathname, if data is not provided. 
    if data is None:
        try: 
            df = pd.read_csv(data_path, parse_dates=True, index_col=data_index_col)

        # Need to improve how exceptions/errors are handled
        except Exception as e: 
            df = pd.read_csv(data_path)
            print_basic_exception(e); print('\n\nFirst 3 rows shown below.')
            return df.head(3)

    # If df provided, try to set index if not already datetime
    else:
        # Make copy bc below uses 'inplace=True' and want to avoid modifying 'data'
        df = data.copy()
        
        # This part needs to be better.. e.g. except: //
        if type(df.index) != pd.DatetimeIndex and df.index.dtype != 'O': #  type(df.index) != str:
            
            # Below needs to be changed..
            try: df.set_index(data_index_col, inplace=True)
            except: df.set_index(df.columns[data_index_col], inplace=True)
                
            df.index = pd.to_datetime(df.index)
            
        else:
            first_idx = df.index[0]
            df.index = pd.to_datetime(df.index)
            print('* Index converted to datetime. Please verify example below.\nfrom {} to {}\n'\
                  .format(first_idx, df.index[0]))
            
    ## Drop duplicates
    if data_drop_duplicates:
    # if len(set(df.index)) != len(df.index):
        df = df.loc[~df.index.duplicated(keep=keep)]
        print('* Dropping duplicate index values. Keeping {}.\n'.format(keep))
    
    # - - - - - - - - - - - - - - - DEL - - - - - - - - - - - - - - - - - - - - - - - - - //
    # Fuzzy match col names to set y,yhat,ci without specifying in params
    # if dumb_y:
        
        ## Below may be better as param later..
        # try:
        # simple_match = {}
        
        # A dumb message to indicate this is not a very good way to do it..
        # dumb_msg = "Moo!!!! * This overrides 'y', 'yhat', 'ci_lower', 'ci_upper'\n"
        # print(dumb_msg)
        
        # Select closest match...
        # y_cand = 
        # yhat_cand = 
        # ci_lower_cand = 
        # ci_upper_cand = 
        # except:
        # try again~
    # else: 
    # - - - - - - - - - - - - - - - DEL - - - - - - - - - - - - - - - - - - - - - - - - -
    
    # Create plots
    fig = plt.figure(figsize=plot_size)

    gs = gridspec.GridSpec(3, 2, height_ratios=[3, 4, 3])
    
    # First plot
    ax1 = fig.add_subplot(gs[0, :])
    
    # Second plot
    ax2 = fig.add_subplot(gs[1, :])
    
    # Lower left
    ax3 = fig.add_subplot(gs[2, 0])
    
    # Lower right
    ax4 = fig.add_subplot(gs[2, 1])
    
    # Plot forecasts without setting style
    with plt.style.context(plot_style):
        
        # if not specified start date is 5 years prior
        if start_date is None: start_date = df.index[-1] - pd.DateOffset(months=60)
        
        ## Subset dataframe based on date <--- this should be changed to be based on k_months
        subset_df = df.pipe(process_date, start_date, end_date, date_format) # process_date(df, )
        
        # Prep to plot in a forloop
        df_all = [df, subset_df]
        top_two_plots = [ax1, ax2]       
        
        for dataframe, axplot in zip(df_all, top_two_plots):
            
            # Plot observed
            dataframe[y].plot(ax=axplot, label='Observed', c=y_color, lw=y_lw,
                             marker=y_marker, markersize=y_markersize, ls=y_ls, 
                              markerfacecolor=y_facecolor)
            
            # Plot predicted
            dataframe[yhat].plot(ax=axplot, label='Predicted', lw=yhat_lw, c=yhat_color,
                                marker=yhat_marker, markersize=yhat_markersize, 
                                 markerfacecolor=yhat_facecolor, ls=yhat_ls)
            
            ## Plot confidence interval
            # if ci_lower and ci_upper is not None:
            try:
                axplot.fill_between(x=dataframe.index, y1=dataframe[ci_lower], y2=dataframe[ci_upper],
                                    alpha=0.2, color=ci_color, label='{} Conf. Interval'.format(ci_label))
            except Exception as e:
                print_basic_exception(e) # <---- need something here
            
            # Plot horizontal line at y=0
            axplot.axhline(y=0, c='gray', alpha=1, ls=':', lw=2)
            
            # Set labels, titles
            axplot.set_ylabel('Annualized rate', size=plot_titlesize-6)
            axplot.set_title('Model Fit/Forecast from {} to {}'\
                             .format(dataframe.index[0].strftime('%B %Y'), dataframe.index[-1].strftime('%B %Y')),
                             size=plot_titlesize)
            
        
        
        ## Plot horizontal line in top two plots if observed contains null
        ## Assuming only null in 'y' is bc no data avail at date. If df has null otherwise, problem.
        ## Is sep. loop bc of axplot.get_ylim()
        for dataframe, axplot in zip(df_all, top_two_plots):
            
            # Check if observed contains null vals
            if dataframe[y].isna().sum() > 0:
                
                # Get last date
                y_end = dataframe[y].dropna().index[-1]
                print(y_end)
                ax_ylim = axplot.get_ylim()[1]

                # If observed end is before end of forecast end
                if y_end < dataframe.index[-1]:
                    # Plot vertical line
                    axplot.axvline(y_end, c=vline_color, ls=vline_ls, lw=vline_lw)
                    # Plot date as text
                    axplot.text(x=y_end, y=ax_ylim * 0.9, s=y_end.strftime('%b %d, %Y'), 
                                ha='right', va='center', size=plot_titlesize-6)
        
        # Calc. residual (not standardized)
        resid = (df[y] - df[yhat]).dropna() 
        
        # Plot residuals in lower left plot
        resid.plot(ax=ax3, ls=':', lw=1, marker='o', alpha=0.75)
        
        # Plot line at y=0
        ax3.axhline(y=0, c='k', ls='--', lw=2, alpha=0.75)
        
        # Set lower left title
        ax3.set_title('Residuals', size=plot_titlesize)
        
        # Plot autocorr in lower right plot
        plot_acf(resid, ax=ax4)
        
        # Set lower right title, x-label
        ax4.set_title('Autocorrelation of Residuals', size=plot_titlesize)
        ax4.set_xlabel('Lag', size=plot_titlesize-6)
        
        # Set xticks, labels in loop <--- this part can be better!! mybad--short on time..
        for f_ax in [ax1, ax2, ax3, ax4]:
            f_ax.tick_params(axis='both', rotation=0, labelsize=plot_titlesize-6)
            f_ax.grid('both', ls=':')
            
        for f_ax in [ax1, ax2, ax3]:
            f_ax.set_xlabel('Date', size=plot_titlesize-6)
        
        for f_ax in [ax1, ax2]:
            f_ax.legend(loc='best')
        
        # Print mse, rmse
        if print_mse:
            mse = np.mean(np.square(resid))
            rmse = np.sqrt(mse)
            
            print('- from {} to {} -'.format(resid.index[0].strftime('%B %d, %Y'),
                                             resid.index[-1].strftime('%B %d, %Y')))
            
            print('mse: {:.4f}\nrmse: {:.4f}'.format(mse, rmse))
        
        plt.tight_layout()
        plt.show()
        return fig
    
    
def process_date(df, start_date=None, end_date=None, date_format='%Y-%m-%d'):
    """Return subset based on start, end date"""
    
    if start_date is not None:
        start_datetime = pd.to_datetime(start_date, format=date_format)
        subset = df[(df.index >= start_datetime)] ## sample.truncate(after=)
        
    if end_date is not None:
        end_datetime = pd.to_datetime(end_date, format=date_format)
        subset = df[(df.index < end_datetime)]
    
    return subset
    # try: // except: # Exception as e:
    
# Below is not very good way of doing this...
def print_basic_exception(e, traceback=False):
    if traceback: raise e
    else: print('* Oops. Something went wrong ---> {}'.format(e))