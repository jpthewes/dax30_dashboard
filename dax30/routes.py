from dax30 import app
import json, plotly
from flask import render_template, request
from wrangling_scripts.wrangle_data import return_figures, return_predicted_figure


@app.errorhandler(RuntimeError)
def handle_exception(e):
    # if explicitly thrown error by us, which occurs when max number of API Calls are reached:
    return render_template("timer.html")


@app.route('/')
@app.route('/index')
def index():

    figures = return_figures()

    # plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('index.html',
                           ids=ids,
                           figuresJSON=figuresJSON)
    

@app.route('/predict')
def predict():
    # fetch from request the days to make predictions for and get specific figure
    n_days = int(request.args.get("n_days"))
    figures = return_predicted_figure(n_days)

    # plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('predict.html', ids=ids,
                           figuresJSON=figuresJSON, predicted_days=n_days)
    
    
@app.route('/predict_select')
def predict_select():
    # template to select the number of predicted days from
    return render_template('predict_select.html')