# dax30_dashboard
This uses data from the stock market via the alphavantage API to create a stock market dashboard. 
It currently only shows the 3 big german car manufacturers DAI, BMW, VW due to free API restrictions but is meant to have the whole DAX 30 companies.
This project was part of the Udacity Data Science Nanodegree.

The dashboard is deployed using Heroku (https://dax30-dashboard.herokuapp.com/) but it can also be tested locally.
For that you just need to comment out [this line](https://github.com/jpthewes/dax30_dashboard/blob/1581d40119a64e87dc6728c17f8432753e34c805/dax30.py#L3) and run:
```python dax30.py```
