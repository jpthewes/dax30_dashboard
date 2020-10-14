#!/usr/bin/env python3
import pandas as pd
import plotly.graph_objs as go
import requests

from time import sleep

#DAX = ["ADS", "ALV", "BAS", "BAYN", "BEI", "BMW", "CON", "1COV", "DAI", "DHER", "DBK", "DB1", "DPW", "DTE", "DWNI", "EOAN","FRE", "FME", "HEI", "HEN3", "IFX", "LIN", "MRK", "MTX", "MUV2", "RWE", "SAP", "SIE", "VOW3", "VNA"]
DAX = ['DAI', 'BMW', 'VOW3']


def get_daily_stock_data(symbol="DAI.DEX"):
    key = 'W27HV1UZPA88SX7K'
    # gets 100 last daily datapoints
    data = requests.get(f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=compact&apikey={key}")

    print(data.json().keys())
    if len(data.json().keys()) == 2:
        df = pd.DataFrame.from_dict(data.json()['Time Series (Daily)'], orient='index')
        print("Succes: ", symbol)
        print(df)
        return df
    else:
        print("Failed to fetch the following data: ", symbol, data.json()['Note'])
        raise RuntimeError("The maximum requests of the API have been reached, please wait for a while and retry!")


def get_dax_data():
    DAX_pre = [element+".DEX" for element in DAX] # add suffix for german trade symbols
    print(DAX_pre)
    DAX_dfs = {}
    for company in DAX_pre:
        DAX_dfs[company] = get_daily_stock_data(company)
        #sleep(12) # API is limited
    return DAX_dfs

def return_figures():
    """Creates four plotly visualizations

    Args:
        None

    Returns:
        list (dict): list containing the four plotly visualizations

    """

    DAX_dfs = get_dax_data()

    
    graph_one = []    

    for i, company in enumerate(DAX_dfs):
      graph_one.append(
          go.Scatter(
          x = DAX_dfs[company].index,
          y = DAX_dfs[company]['4. close'],
          mode = 'lines',
          name = company
          )
      )

    layout_one = dict(title = 'Chart Perfomance over the last 100 days',
                xaxis = dict(title = 'days'),
                yaxis = dict(title = 'Value in €'),
                )

# second chart plots ararble land for 2015 as a bar chart    
    graph_two = []

    for i, company in enumerate(DAX_dfs):
      graph_two.append(
          go.Scatter(
          x = DAX_dfs[company].index,
          y = 100*(DAX_dfs[company]['4. close'].astype('float64') - DAX_dfs[company]['4. close'].astype('float64').iloc[0])/DAX_dfs[company]['4. close'].astype('float64').iloc[0],
          mode = 'lines',
          name = company
          )
      )

    layout_two = dict(title = 'Relative Chart Perfomance over the last 100 days',
                xaxis = dict(title = 'days'),
                yaxis = dict(title = 'percent'),
                )


# third chart plots percent of population that is rural from 1990 to 2015
    graph_three = []
    for i, company in enumerate(DAX_dfs):
      graph_three.append(
          go.Box(
          y = DAX_dfs[company]['5. volume'].astype('float64'),
          name = company
          )
      )

    layout_three = dict(title = 'Trade Volume over the last 100 days',
                xaxis = dict(title = 'days'),
                yaxis = dict(title = 'Trade Volume in €'),
                )
    
# fourth chart shows rural population vs arable land
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
    layout_four = dict(title = 'Relative interday chart volatilty in the last 100 days (absolute)',
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

#return_figures()