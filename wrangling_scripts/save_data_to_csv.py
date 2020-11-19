from alpha_vantage.timeseries import TimeSeries
import json
import argparse
import os


def save_data_to_csv(symbol):
    ''' Fetches data from the Alphavantage API and saves it into csv file.

    Args:
        symbol: stock symbol to fetch data for
        
    Returns:
        None
    '''
    # inspired by: https://github.com/yacoubb/stock-trading-ml
    credentials = json.load(open('creds.json', 'r'))
    api_key = credentials['av_api_key']
    

    # get data from API
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, _ = ts.get_daily(symbol, outputsize='full')
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    target_path = os.path.join(script_dir, f'csv_data/{symbol}_daily.csv')
    
    #user debug output
    print(symbol)
    print(data.head(10))
    print("Saving to ", target_path)
    
    # Save to csv
    data.to_csv(target_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('symbol', type=str, help="the stock symbol you want to download")

    namespace = parser.parse_args()
    save_data_to_csv(**vars(namespace))
