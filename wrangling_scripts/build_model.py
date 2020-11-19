import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Activation
from tensorflow.keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import os
import argparse


script_dir = os.path.dirname(os.path.realpath(__file__))
# Parameters:
history_points = 50
CSV_PATH = os.path.join(script_dir, 'csv_data/DAI.DEX_daily.csv')
MODEL_PRE = os.path.join(script_dir, '../models/basic_model_')
NUM_EPOCHS = 10#0
CUTOFF_LAST_N_DAYS = 300
TEST_SPLIT = 0.9


def shift(arr, num, fill_value=np.nan):
    ''' Shifts an array by a number of steps and fills the remaining spots.
    
    Args:
        arr: array to be shifted
        num: number of places to shift
        fill_value: value to fill into free spots
        
    Returns:
        result: shifted array
    '''
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def clean_and_normalise(df, override_normaliser=False):
    ''' 
    Cleans the given dataframe in predefined steps and normalizes the data
    either based on the given dataframe or the original dataframe trained on.
    
    Args:
        df: Dataframe to be modified
        override_normaliser: flag to decide if normalise based on the given 
                             dataframe or the original dataframe trained on

    Returns:
        data: cleaned data
        normalised_data: cleaned AND normalised data
    '''
    if override_normaliser:
        df = df.drop('date', axis=1)
    df = df.iloc[::-1].reset_index(drop=True)
    df = df.drop(0, axis=0)

    data = df.values
    if override_normaliser:
        data_normaliser = preprocessing.MinMaxScaler()
        data_normalised = data_normaliser.fit_transform(data)
    else: 
        df = pd.read_csv(CSV_PATH)
        df = df.drop('date', axis=1)
        df = df.iloc[::-1].reset_index(drop=True)
        df = df.drop(0, axis=0)

        training_data = df.values
        
        data_normaliser = preprocessing.MinMaxScaler()
        data_normaliser.fit(training_data)
        data_normalised = data_normaliser.transform(data)
        
        
    return data, data_normalised

def csv_to_dataset(csv_path, predicted_days):
    ''' 
    Reads in a csv file and creates the arrays needed for model training 
    and evaluation from that. 
    
    
    Args:
        csv_path: filepath to load the csv file from
        
    Returns:
        stock_histories_normalised: x values for prediction
        next_days_open_values_normalised: y values which were normalised 
        next_days_open_values: y values in original dataspan
        y_normaliser: normaliser which was used for normalizing the data
    '''
    # inspired by: https://github.com/yacoubb/stock-trading-ml, but adapted for many days in advance instead of 1 day
    
    df = pd.read_csv(csv_path)
    cleaned_data, data_normalised = clean_and_normalise(df, override_normaliser=True)
    # prepare the model input, the stock histories
    stock_histories_normalised = np.array([data_normalised[i - history_points:i].copy() for i in range(history_points,len(data_normalised) - predicted_days)])
    # normalized y data
    next_days_open_values_normalised = list()
    for i in range(history_points,len(data_normalised) - predicted_days):
        next_days = []
        for k in range(1, predicted_days + 1): # how many days in advance to predict
            next_days.append(data_normalised[i + k, 0].copy())
        next_days_open_values_normalised.append(next_days)
    next_days_open_values_normalised = np.array(next_days_open_values_normalised)

    # original data values for y:
    next_days_open_values = list()
    for i in range(history_points,len(cleaned_data) - predicted_days):
        next_days = []
        for k in range(1, predicted_days + 1): # how many days in advance to predict
            next_days.append(cleaned_data[i + k, 0].copy())
        next_days_open_values.append(next_days)
    next_days_open_values = np.array(next_days_open_values)
       
    y_normaliser = preprocessing.MinMaxScaler()
    y_normaliser.fit(next_days_open_values)
    
    assert stock_histories_normalised.shape[0] == next_days_open_values_normalised.shape[0]
    return stock_histories_normalised[:-CUTOFF_LAST_N_DAYS], next_days_open_values_normalised[:-CUTOFF_LAST_N_DAYS], next_days_open_values[:-CUTOFF_LAST_N_DAYS], y_normaliser


def plot_test_results(unscaled_y_test, y_test_predicted, predicted_days, title, mse, save=False, train=False):
    ''' Plots the predicted and actual values side to side in a graph. 
    
    Args: 
        unscaled_y_test: unscaled y data of the test split, true values from data
        y_test_predicted: y values which were predicted by the model and scaled back to original dataspan
        predicted_days: days in the future which were predicted
        title: title of plot
        mse: mean squared error of prediction
        
    Returns:
        None
    '''
    
    plt.clf()
    plt.plot(unscaled_y_test, label='real')
    plt.plot(y_test_predicted[:, 0], label='predicted')
    # shift the last prediction which is predicted_days ahead the same amount into the future/to the right to vizualise how it is predicted
    # fill up with the first predicted value (y_test_predicted[0, predicted_days-1])
    pred_last_shifted = shift(y_test_predicted[:, predicted_days-1], predicted_days, y_test_predicted[0, predicted_days-1])
    plt.plot(pred_last_shifted, label='predicted_last')

    plt.legend(['Real', 'Predicted', f'Predicted {predicted_days} days in advance'])
    plt.title(title)
    plt.text(0.05*len(unscaled_y_test), np.min(unscaled_y_test), f'Scaled MSE: {mse}')

    if save:
        filename = f'../pictures/{predicted_days}days_ahead_train.png' if train else f'../pictures/{predicted_days}days_ahead.png'
        plt.savefig(os.path.join(script_dir, filename))
    else:
        plt.show()
    
    


def build_model(predicted_days, save_fig=False):
    ''' 
    Creates, trains, evaluates and saves the ML model used for stock prediction.
    The model is based on some parameters above such as the number of predicted
    days and the number of epochs to be trained on.
    
    Args:
        None
    
    Returns:
        None
    '''
    # workflow is inspired by https://github.com/yacoubb/stock-trading-ml, but different data split, model architecture and parameters are used
    #get data from csv
    stock_histories, next_days_open_values, unscaled_y, y_normaliser = csv_to_dataset(CSV_PATH, predicted_days)

    # split into train and test data based on test_split param 
    n = int(stock_histories.shape[0] * TEST_SPLIT)
    x_train, x_test = np.split(stock_histories, [n])
    y_train, y_test = np.split(next_days_open_values, [n])
    unscaled_y_train, unscaled_y_test = np.split(unscaled_y[:,0], [n])

    # model architecture, tried different architectures and found this to be most accurate
    input_layer = Input(shape=(history_points, 5))
    x = LSTM(50, return_sequences=True)(input_layer)
    x = Dropout(0.15)(x)
    x = LSTM(21)(x)
    x = Dropout(0.1)(x)
    x = Dense(128)(x)
    x = Dense(64)(x)
    output = Dense(predicted_days, activation="linear")(x)

    model = Model(inputs=input_layer, outputs=output)
    # adam was more accurate that nadam, rmsprob, adadelta
    adam = optimizers.Adam(lr=0.0005)
    model.compile(optimizer=adam, loss='mse')
    print(model.summary())
    model.fit(x=x_train, y=y_train, batch_size=80, epochs=NUM_EPOCHS, shuffle=True, validation_split=0.1)


    # prediction
    y_test_predicted_norm = model.predict(x_test)
    y_test_predicted = y_normaliser.inverse_transform(y_test_predicted_norm)
    y_predicted_norm = model.predict(stock_histories)
    y_predicted = y_normaliser.inverse_transform(y_predicted_norm)
    
    # evaluation with MSE and plots
    mse = np.mean(np.square(unscaled_y_test - y_test_predicted[:,0]))
    scaled_mse = mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
    print("Scaled MSE: ", scaled_mse)
    
    plot_test_results(unscaled_y_test, y_test_predicted, predicted_days, "test data evaluation", scaled_mse, save=save_fig)

    plot_test_results(unscaled_y[:, 0], y_predicted, predicted_days, "training and test data", scaled_mse, save=save_fig, train=True)
    
    #save model
    model_path = f"{MODEL_PRE}{predicted_days}.h5"
    model.save(model_path)

def load_model_predict(histories, predicted_days):
    '''
    Loads the fitting model for the predicted days and predicts bases on 
    the given data points (histories).
    
    Args: 
        histories: x values for the prediction, the data of the past n days of stock data
        predicted_days: the number of days to predict into the future
    
    Returns:
        predicted: predicted y data
    '''
    model_path = f"{MODEL_PRE}{predicted_days}.h5"
    model = load_model(model_path, compile=False)
    predicted = model.predict(histories)
    return predicted
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--predicted_days', type=int, help="the number of days to predict into the future", required=False)
    parser.add_argument('-a', dest="train_all", action="store_true", help="run trough all predicted_days from 1-10 and save figures")

    args = parser.parse_args()
    if args.train_all:
        for i in range(1,11): #1-10
            build_model(i, save_fig=True)
    else:
        if args.predicted_days < 1:
            print("Please provide the number of predictes days as positional argument!")
            exit(1)
        build_model(args.predicted_days)