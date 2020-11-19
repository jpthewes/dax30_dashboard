# dax30_dashboard

## Project Definition:
### Project Overview:
This project uses data from the stock market via the alphavantage API to create a stock market dashboard. 
It currently shows the 3 big german car manufacturers DAI, BMW, VW due to free API restrictions but is meant to have the whole DAX 30 companies. Moreover to just showing the historical data of the stocks, the project also includes a machine learning approach to predict the future stock prices based on the recent stock activity.
This project was part of the Udacity Data Science Nanodegree.

### Problem Statement:
Many individuals have been recently started stock trading for individual investement. A big issue for these people is to find valuable information for trading which helps them to assess the stocks and try to predict whether they will rise or fall in value. 
This project aims to assist these individuals with a machine learning approach. **The goal of this project is clearly not to give buy or sell recommendations** but rather to test how well ML can make those predictions and provide more information to investors. 

### Metrics:
The accuracy of the predictions will be assessed using visual comparison of stock charts and MSE (= mean square error). Visual comparison is seen as valuable here because the development of the value over time can most easily be assessed by a human instead of comprimising this information into one number. On the other hand a numerical metric is necessary to give an overall idea while training and tuning the model. Therefore MSE is used.

## Data Analysis (Exploration/Visualization):
The data which is used was fetched using the [Alphavantage API](https://www.alphavantage.co/). The training of the model was performed with data from 2005-01-03 until 2020-11-13. The data was explored using pandas and matplotlib in the development process. Fortunately the data which is pulled from the API is already clean and can be used directly. Continously the most recent data is vizualised at the [landing page](https://dax30-dashboard.herokuapp.com/) of the web app.

## Methodology:
### Data Preprocessing:
The data is already fairly clean and only needs to be brought into the right format using pandas. This is performed in the function csv_to_dataset() in build_model.py.

### Implementation/Refinement:
I chose a model architecture which uses LSTM cells as a main part. Those LSTM(=Long Short Term Memory) cells have been shown to provide promising results when predicting time-based data, because LSTM as a kind of Recurrent Neural Network (RNN) use previous time events to be informed about later ones. Different than other RNNs, LSTM also uses long term dependencies for prediction. For more information please take a look [here](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).
#### Model Architecture:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 50, 5)]           0         
_________________________________________________________________
lstm (LSTM)                  (None, 50, 50)            11200     
_________________________________________________________________
dropout (Dropout)            (None, 50, 50)            0         
_________________________________________________________________
lstm_1 (LSTM)                (None, 21)                6048      
_________________________________________________________________
dropout_1 (Dropout)          (None, 21)                0         
_________________________________________________________________
dense (Dense)                (None, 128)               2816      
_________________________________________________________________
dense_1 (Dense)              (None, 64)                8256      
_________________________________________________________________
dense_2 (Dense)              (None, 11)                715       
=================================================================
Total params: 29,035
Trainable params: 29,035
Non-trainable params: 0
```

For each prediction the model gets an array of shape (NUM_PREDICTIONS, 50, 5). This correlates to the last 50 timestamps of stock data with 5 data points for each timestamp as such:
```
            1. open  2. high   3. low 4. close 5. volume
2020-11-17  53.7400  54.5400  53.2800  54.5400   3717551
2020-11-16  53.0000  53.9500  52.5300  53.7800   4494536
2020-11-13  51.8500  52.7400  51.7400  52.5000   3168646
2020-11-12  52.3000  53.0600  51.8100  52.2000   3324412
2020-11-11  52.2100  53.4500  52.1900  52.9900   4066917
...
```
The output depends on the number of days to predict into the future and is of shape (NUM_PREDICTIONS, 1, DAYS_IN_FUTURE).

Parameters which can be adjusted while building the model are:
- history_points: based on how many days in the past the predicted is performed
- predicted_days: how many days in the future to predict
- NUM_EPOCHS: number of epochs to train the model
- CUTOFF_LAST_N_DAYS: number of days to cut off at the end of the fetched data (will be explained later)

Furthermore, in the process of training the models the parameters epoch_size and learning rate were tweaked and the model architecture was adjusted with the help of the above mentioned metrics.

## Results:
The prediction of the next days of stock prices is now performed on a different model based on the number of days to predict. This is necessary, because the models need to have fixed output layers. Therefore one needs to select the number of days before seing the predictions: https://dax30-dashboard.herokuapp.com/predict_select. Based on this user input the corresponding model is chosen and predictions are performed into the future based on most recent data pulled from the API (e.g.: https://dax30-dashboard.herokuapp.com/predict?n_days=10). 
Due to API limitations of 5 possible calls per minute a waiting [timer](https://github.com/jpthewes/dax30_dashboard/blob/master/dax30/templates/timer.html) has been introduced to inform about this limitation. This can be tested when refreshing the page 2 times in 1 minute.

In addition to this prediction feature, some key facts about the stocks are displayed at the [landing page](https://dax30-dashboard.herokuapp.com/) of the web app.

The scaled MSE across all models are below 10$ which I consider acceptable considering the simple model and the limited amount of information. The MSE for each model is plotted in the [visualization](https://github.com/jpthewes/dax30_dashboard/tree/master/pictures).

## Conclusion:
### Reflection:
While validating the model using the mentioned metrics RMSE and visualizations, it became apparent that the more days in the future I wanted to predict, the less accurate the predictions were. This seems plausible. But if you take a look at the [pictures](https://github.com/jpthewes/dax30_dashboard/tree/master/pictures), it becomes visible that if you train a model for predicting e.g. 10 days into the future also the prediction of 1 day into the future gets worse. This is to be discussed in the [improvements section](#Instructions:).

What also comes apparent when looking at the images is, that the predictions somehow are a little lower and behind the more predictions are performed. This could be related to the long term memory character of LSTM.

Concerning the turbulences at the stock market in the current year 2020 due to covid, the model parameter CUTOFF_LAST_N_DAYS was introduced for model training and validation to avoid a false negative of an otherwise adequate model.

One aspect which should also be noted is, that the predictions here are solely based on historical data and do not take into concern what is currently being published in the news about the company or how good or bad their fundamental stats (EBIT etc.) are. Therefore it is expected to be not very accurate but I'm happy that at least the trends of the stocks are being followed. 


### Improvements:
Improvements which can be implemented are:
- train the model on more data and on more than one stock
- enrich the input data with more data (e.g. indicators)
- tune hyperparameters automatically or manually

Moreover it seems promising to use many models for predicting one day each instead of using one model to predict all the future days. Therefore the output shape of the model would not anymore be (NUM_PREDICTIONS, 1, DAYS_IN_FUTURE) but rather (NUM_PREDICTIONS, 1). Each model would predict only one day but it would be specialized to predict n days ahead. We would then need n number of models (n=DAYS_IN_FUTURE). 



## Instructions:
The dashboard is deployed online using Heroku (https://dax30-dashboard.herokuapp.com/) but it can also be tested locally.
For that you just need to comment out [this line](https://github.com/jpthewes/dax30_dashboard/blob/04c67dc7844b55e76fcfe6b789c94c0970ddda40/dax30.py#L3) and run:
```python3 dax30.py```
### Packages:
The required python3 packages are defined in the requirements.txt and can be installed using 
```pip3 install -r requirements.txt```