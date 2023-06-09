""" Functions used for preprocessing data and training, testing model."""
import os
import pickle
import numpy as np
import pandas as pd
import re
import time

from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import plot_confusion_matrix

from document_classifier.core.definitions import ModelRegressor


def preprocess_text(text: str) -> str:
    """ Remove special characters, punctuations, etc.
    Args:
        text: input text

    Returns:
        text: preprocessed text (removed puncutations, numbers)
    """
    text = text.lower()
    remove_numbers = re.compile('[0-9]+')
    remove_special_characters = re.compile('[^A-Za-z0-9]+')

    text = re.sub(remove_numbers, ' ', text)
    text = re.sub(remove_special_characters, ' ', text)
    return text.strip()


def compute_tfidf(corpus: pd.Series,
                  stop_words='english',
                  ngram_range=(1, 1),
                  max_features=None):
    """ Convert text to a matrix of TF-IDF features.
    Args:
        corpus: input content
        stop_words: Words in a stop list which are filtered out before or after processing of
        natural language data.
        ngram_range: The lower and upper boundary of the range of n-values for different
        n-grams to be extracted.
        max_features: If not None, build a vocabulary that only consider the top max_features
        ordered by term frequency across the corpus

    Returns:
        X: matrix of TF-IDF features
        vectorizer: TfidfVectorizer object
    """
    vectorizer = TfidfVectorizer(
        input='content',
        stop_words=stop_words,
        ngram_range=ngram_range,
        min_df=3,
        max_df=0.9,
        max_features=max_features
    )
    print('Computing tfidf features...', end='')
    vectorizer.fit(corpus)
    tfidf_matrix = vectorizer.transform(corpus)
    print('done!')
    return tfidf_matrix, vectorizer


def encode_labels(labels: list, data: pd.Series) -> pd.Series:
    """ Encode labels into numbers.
    Args:
        labels: labels for classes
        data: target data

    Returns
        result: encoded target data
    """
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(labels)
    result = label_encoder.transform(data)
    return result


def get_model(model: ModelRegressor, X_train: np.ndarray, y_train: np.ndarray, file_name: str):
    """ Get model, if model exists, load model. Otherwise, train a new model.
    """
    if os.path.exists(file_name):
        model = pickle.load(open(file_name, 'rb'))
        training_time = 0
        return model, training_time
    print('Start training...', end='')
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    print('done!')
    print('Start testing...', end='')
    training_time = end_time - start_time
    pickle.dump(model, open(file_name, 'wb'))
    return model, training_time


def evaluate_mode(model: ModelRegressor, X_test, y_test, labels):
    """ Evaluate model performance"""

    predictions = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    metrics_report = classification_report(y_test, predictions, target_names=labels)
    precision, recall, fscore, train_support = score(y_test, predictions, average='weighted')
    return predictions, accuracy, metrics_report, (precision, recall, fscore)


def show_model_performance(model, accuracy, metrics_report, training_time,  X_test, y_test, labels):
    """ Show model performance"""
    print('Total time: {:.2f}s'.format(training_time))
    print('accuracy: {}'.format(accuracy))
    print('=' * 100)
    print(metrics_report)
    plot_confusion_matrix(model, X_test, y_test, display_labels=labels, xticks_rotation='vertical', cmap="BuPu")
