#!/usr/bin/env python3
import pandas as pd
import plotly.graph_objs as go
import requests
import numpy as np
import datetime
import json
import os
from time import sleep
from sklearn import preprocessing
from backports.datetime_fromisoformat import MonkeyPatch
MonkeyPatch.patch_fromisoformat()

from flask import render_template

from .basic_model import load_model_predict, history_points, clean_and_normalise

#Parameters
LAST_N_DAYS_HISTORY = 50

#DAX = ["ADS", "ALV", "BAS", "BAYN", "BEI", "BMW", "CON", "1COV", "DAI", "DHER", "DBK", "DB1", "DPW", "DTE", "DWNI", "EOAN","FRE", "FME", "HEI", "HEN3", "IFX", "LIN", "MRK", "MTX", "MUV2", "RWE", "SAP", "SIE", "VOW3", "VNA"]
DAX = ['DAI', 'BMW', 'VOW3']


def get_daily_stock_data(symbol):
    ''' Fetches daily stock data for the given stock symbol and return the df. 
    
    Args:
        symbol: stock symbol to fetch data from
        
    Returns:
        df: Dataframe with the stock data
    '''
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    credentials = json.load(open(os.path.join(script_dir, 'creds.json'), 'r'))
    key = credentials['av_api_key']
    # gets last daily datapoints
    data = requests.get(f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=compact&apikey={key}")

    #print(data.json().keys())
    if len(data.json().keys()) == 2:
        df = pd.DataFrame.from_dict(data.json()['Time Series (Daily)'], orient='index').sort_index(ascending=False)
        print("Succes: ", symbol)
        return df[:LAST_N_DAYS_HISTORY+1]
    else:
        print("Failed to fetch the following data: ", symbol, data.json()['Note'])
        raise RuntimeError("The maximum requests of the API have been reached, please wait for a while and retry!")


def get_dax_data():
    ''' Fetches all the stock data for the defined companies in the parameters
    above. Returns a dict of Dataframes for all the companies each.
    
    Args:
        None
    
    Returns: 
        DAX_dfs (dict): Dict which contains the related dataframe for each company
    '''
    DAX_pre = [element+".DEX" for element in DAX] # add suffix for german trade symbols
    print(DAX_pre)
    DAX_dfs = {}
    for company in DAX_pre:
        DAX_dfs[company] = get_daily_stock_data(company)
        #sleep(12) # API is limited
    return DAX_dfs

def return_predicted_figure(n_days):
    ''' 
    Creates the figures to be vizualised on the web page for the number of 
    predicted days given. For that the data is fetched and prepared, then the 
    predictions are made and the predictions are packed in graphs.
    
    Args:
        n_days (int): number of days to be predicted for

    Returns:
        figures: list containing the plotly visualization
    '''
    print(f"creating figures for predictions of next {n_days}")
    DAX_dfs = get_dax_data()
    
    graph_one = []

    # predict data create graph for it
    for i, company in enumerate(DAX_dfs):
        # predict and append
        past_data = DAX_dfs[company]
        cleaned_data, normalized_data = clean_and_normalise(past_data[-history_points-1:])
        normalized_data = np.expand_dims(normalized_data, axis=0)
        
        # get open values in necessary output shape to inverse transform the predictions later
        open_values = list()
        for i in range(len(cleaned_data)):
            next_days = []
            for k in range(0, n_days): 
                next_days.append(cleaned_data[i, 0])
            open_values.append(next_days)
        open_values = np.array(open_values)
        # fit normalizer with those values
        y_normaliser = preprocessing.MinMaxScaler()
        y_normaliser.fit(open_values)

        # actual prediction 
        pred = load_model_predict(normalized_data, predicted_days=n_days)
        pred_scaled = y_normaliser.inverse_transform(pred)
        #hist_and_pred = np.concatenate((DAX_dfs[company]['1. open'], pred_scaled), axis=0) #append complete last row (=future) to predictions
        
        # append last prediction (= future values) to the 
        dates_index = []
        for days_delta in range(1, n_days+1):
            n_days_ahead = datetime.date.fromisoformat(DAX_dfs[company].index[0]) + pd.tseries.offsets.BDay(days_delta)
            dates_index.append(n_days_ahead.isoformat().split('T')[0])

        # create own graoh for pred
        graph_one.append(
            go.Scatter(
            x = dates_index,
            y = pred_scaled[0],
            mode = 'lines',
            name = f"Prediction {company}"
            )
        )
        
        
    # graphs for past real data
    for i, company in enumerate(DAX_dfs):
        graph_one.append(
          go.Scatter(
          x = DAX_dfs[company].index,
          y = DAX_dfs[company]['1. open'],
          mode = 'lines',
          name = company
          )
        )
      
    layout_one = dict(title = f'Chart Perfomance over the last {LAST_N_DAYS_HISTORY} days',
            xaxis = dict(title = 'days'),
            yaxis = dict(title = 'Value in €'),
            )
    
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    
    return figures

    
def return_figures():
    """Creates four plotly visualizations for the DAX stock data:
    - absolute chart performance
    - relative chart performance
    - trade volume
    - relative interday chart volatilty

    Args:
        None

    Returns:
        figures: list containing the four plotly visualizations

    """

    DAX_dfs = get_dax_data()

    
    graph_one = []    

    for i, company in enumerate(DAX_dfs):
      graph_one.append(
          go.Scatter(
          x = DAX_dfs[company].index,
          y = DAX_dfs[company]['1. open'],
          mode = 'lines',
          name = company
          )
      )

    layout_one = dict(title = f'Chart Perfomance over the last {LAST_N_DAYS_HISTORY} days',
                xaxis = dict(title = 'days'),
                yaxis = dict(title = 'Value in €'),
                )

  
    graph_two = []

    for i, company in enumerate(DAX_dfs):
      graph_two.append(
          go.Scatter(
          x = DAX_dfs[company].index,
          y = 100*(DAX_dfs[company]['1. open'].astype('float64') - DAX_dfs[company]['1. open'].astype('float64').iloc[0])/DAX_dfs[company]['1. open'].astype('float64').iloc[0],
          mode = 'lines',
          name = company
          )
      )

    layout_two = dict(title = f'Relative Chart Perfomance over the last {LAST_N_DAYS_HISTORY} days',
                xaxis = dict(title = 'days'),
                yaxis = dict(title = 'percent'),
                )



    graph_three = []
    for i, company in enumerate(DAX_dfs):
      graph_three.append(
          go.Box(
          y = DAX_dfs[company]['5. volume'].astype('float64'),
          name = company
          )
      )

    layout_three = dict(title = f'Trade Volume over the last {LAST_N_DAYS_HISTORY} days',
                xaxis = dict(title = 'days'),
                yaxis = dict(title = 'Trade Volume in €'),
                )
    

    graph_four = []
    
    #for i, company in enumerate(DAX_dfs):
     # graph_four.append(
      #    go.Scatter(
       #   x = DAX_dfs[company].index,
        #  y = 100*abs(DAX_dfs[company]['4. close'].astype('float64') - DAX_dfs[company]['1. open'].astype('float64'))/DAX_dfs[company]['4. close'].astype('float64').iloc[0],
         # mode = 'lines',
          #name = company,
          #line = {'shape': 'spline', 'smoothing': 1.9}
          #)
      #)

    for i, company in enumerate(DAX_dfs):
      graph_four.append(
          go.Box(
          y = 100*abs(DAX_dfs[company]['4. close'].astype('float64') - DAX_dfs[company]['1. open'].astype('float64'))/DAX_dfs[company]['4. close'].astype('float64').iloc[0],
          name = company
          )
      )
    layout_four = dict(title = f'Relative interday chart volatilty in the last {LAST_N_DAYS_HISTORY} days (absolute)',
                xaxis = dict(title = 'days'),
                yaxis = dict(title = 'percent'),
                )
    
    # append all charts to the figures list
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    figures.append(dict(data=graph_three, layout=layout_three))
    figures.append(dict(data=graph_four, layout=layout_four))

    return figures
