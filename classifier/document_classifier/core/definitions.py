""" This module contains parameter definitions."""
import os
from typing import Union

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

ModelRegressor = Union[LinearRegression, RandomForestRegressor, GradientBoostingRegressor, None]
PATH_ROOT = os.path.dirname(os.path.split(os.path.dirname(__file__))[0])
PATH_DATA_DIRECTORY = os.path.join(PATH_ROOT, "data")
PATH_NEWS = os.path.join(PATH_DATA_DIRECTORY, 'NewsAggregatorDataset/newsCorpora.csv')

LOG_MODEL_NAME = os.path.join(PATH_DATA_DIRECTORY, 'models/logistic_regression.sav')
COLUMN_NAMES = ['id', 'title', 'url', 'pulisher', 'category', 'story', 'hostname', 'timestep']

# parameters of TfidfVectorizer
stop_words = 'english'
ngram_range = (1, 1)
max_features = None
