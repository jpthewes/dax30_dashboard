from flask import Flask

app = Flask(__name__)

from dax30 import routes
