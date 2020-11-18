import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Activation
from tensorflow.keras import optimizers
import numpy as np
np.random.seed(4)
from tensorflow.compat.v1 import set_random_seed
set_random_seed(4)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import os


script_dir = os.path.dirname(os.path.realpath(__file__))
# Parameters:
history_points = 50
predicted_days = 1
CSV_PATH = os.path.join(script_dir, 'csv_data/DAI.DEX_daily.csv')
MODEL_PRE = os.path.join(script_dir, '../models/basic_model_')
MODEL_PATH = f'{MODEL_PRE}{predicted_days}.h5'
NUM_EPOCHS = 100
CUTOFF_LAST_N_DAYS = 300


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

def csv_to_dataset(csv_path):
    ''' 
    Reads in a csv file and creates the arrays needed for model training 
    and evaluation from that. 
    
    Args:
        csv_path: filepath to load the csv file from
        
    Returns:
        ohlcv_histories_normalised: x values for prediction
        technical_indicators_normalised: currently not used
        next_days_open_values_normalised: y values which were normalised 
        next_days_open_values: y values in original dataspan
        y_normaliser: normaliser which was used for normalizing the data
    '''
    df = pd.read_csv(csv_path)
    cleaned_data, data_normalised = clean_and_normalise(df, override_normaliser=True)
    # using the last {history_points} open close high low volume data points, predict the next open value
    ohlcv_histories_normalised = np.array([data_normalised[i - history_points:i].copy() for i in range(history_points,len(data_normalised) - predicted_days)])
    # normalized data
    next_days_open_values_normalised = list()
    for i in range(history_points,len(data_normalised) - predicted_days):
        next_days = []
        for k in range(1, predicted_days + 1): # how many days in advance to predict
            next_days.append(data_normalised[i + k, 0].copy())
        next_days_open_values_normalised.append(next_days)
    next_days_open_values_normalised = np.array(next_days_open_values_normalised)

    # original data values:
    next_days_open_values= list()
    for i in range(history_points,len(cleaned_data) - predicted_days):
        next_days = []
        for k in range(1, predicted_days + 1): # how many days in advance to predict
            next_days.append(cleaned_data[i + k, 0].copy())
        next_days_open_values.append(next_days)
    next_days_open_values = np.array(next_days_open_values)
       
    y_normaliser = preprocessing.MinMaxScaler()
    y_normaliser.fit(next_days_open_values)

    def calc_ema(values, time_period):
        # https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
        sma = np.mean(values[:, 3])
        ema_values = [sma]
        k = 2 / (1 + time_period)
        for i in range(len(his) - time_period, len(his)):
            close = his[i][3]
            ema_values.append(close * k + ema_values[-1] * (1 - k))
        return ema_values[-1]

    technical_indicators = []
    for his in ohlcv_histories_normalised:
        # note since we are using his[3] we are taking the SMA of the closing price
        sma = np.mean(his[:, 3])
        macd = calc_ema(his, 12) - calc_ema(his, 26)
        technical_indicators.append(np.array([sma]))
        # technical_indicators.append(np.array([sma,macd,]))

    technical_indicators = np.array(technical_indicators)

    tech_ind_scaler = preprocessing.MinMaxScaler()
    technical_indicators_normalised = tech_ind_scaler.fit_transform(technical_indicators)
    #print("ASSERT THIS: ", ohlcv_histories_normalised.shape, "EQUALS ====", next_days_open_values_normalised.shape)
    
    assert ohlcv_histories_normalised.shape[0] == next_days_open_values_normalised.shape[0]# == technical_indicators_normalised.shape[0]
    return ohlcv_histories_normalised[:-CUTOFF_LAST_N_DAYS], technical_indicators_normalised[:-CUTOFF_LAST_N_DAYS], next_days_open_values_normalised[:-CUTOFF_LAST_N_DAYS], next_days_open_values[:-CUTOFF_LAST_N_DAYS], y_normaliser


def plot_test_results(unscaled_y_test, y_test_predicted):
    ''' Plots the predicted and actual values side to side in a graph. 
    
    Args: 
        unscaled_y_test: unscaled y data of the test split, true values from data
        y_test_predicted: y values which were predicted by the model and scaled back to original dataspan
        
    Returns:
        None
    '''
    plt.gcf().set_size_inches(22, 15, forward=True)

    start = 0
    end = -1

    real = plt.plot(unscaled_y_test[start:end], label='real')
    pred = plt.plot(y_test_predicted[start:end, 0], label='predicted')
    pred_last_shifted = shift(y_test_predicted[start:end, predicted_days-1], predicted_days, y_test_predicted[start, predicted_days-1])
    pred_last = plt.plot(pred_last_shifted, label='predicted_last')

    plt.legend(['Real', 'Predicted', f'Predicted {predicted_days} in advance'])

    plt.show()

# much credits to: https://github.com/yacoubb/stock-trading-ml
def build_model():
    ''' 
    Creates, trains, evaluates and saves the ML model used for stock prediction.
    The model is based on some parameters above such as the number of predicted
    days and the number of epochs to be trained on.
    
    Args:
        None
    
    Returns:
        None
    '''
    
    ohlcv_histories, _, next_days_open_values, unscaled_y, y_normaliser = csv_to_dataset(CSV_PATH)

    test_split = 0.9
    n = int(ohlcv_histories.shape[0] * test_split)

    ohlcv_train = ohlcv_histories[:n]
    y_train = next_days_open_values[:n]

    ohlcv_test = ohlcv_histories[n:]
    y_test = next_days_open_values[n:]

    #print("UNSCALED Y:", unscaled_y)
    unscaled_y_test = unscaled_y[n:, 0]

    print(ohlcv_train.shape)
    print(ohlcv_test.shape)


    # model architecture

    lstm_input = Input(shape=(history_points, 5), name='lstm_input')
    x = LSTM(21, name='lstm_0')(lstm_input)
    #x = LSTM(50, name='lstm_1')(x)
    x = Dropout(0.05, name='lstm_dropout_0')(x)
    x = Dense(64, name='dense_0')(x)
    x = Activation('sigmoid', name='sigmoid_0')(x)
    output = Dense(predicted_days, name='dense_1_output', activation="linear")(x)
    #output = Activation('linear', name='linear_output')(x)

    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam(lr=0.0005)
    model.compile(optimizer=adam, loss='mse')
    print(model.summary())
    model.fit(x=ohlcv_train, y=y_train, batch_size=80, epochs=NUM_EPOCHS, shuffle=True, validation_split=0.1)


    # evaluation

    y_test_predicted = model.predict(ohlcv_test)
    y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
    y_predicted = model.predict(ohlcv_histories)
    y_predicted = y_normaliser.inverse_transform(y_predicted)

    #print("ASSERT THIS: ", unscaled_y_test.shape, "EQUALS ====", y_test_predicted.shape)
    assert unscaled_y_test.shape[0] == y_test_predicted.shape[0]
    real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted[:,0]))
    scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
    print("Scaled MSE: ", scaled_mse)
    
    model.save(MODEL_PATH)
    
    plot_test_results(unscaled_y_test, y_test_predicted)

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
    build_model()