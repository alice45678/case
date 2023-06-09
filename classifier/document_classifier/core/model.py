""" This module trains logistic regression model and saves the model"""
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import tensorflow_data_validation as tfdv

from document_classifier.core.definitions import PATH_NEWS, COLUMN_NAMES, LOG_MODEL_NAME, stop_words, ngram_range, \
    max_features
from document_classifier.core.functions import preprocess_text, compute_tfidf, encode_labels, \
    get_model, evaluate_mode, show_model_performance

if __name__ == "__main__":
    # load data
    df_news = pd.read_csv(PATH_NEWS, sep='\t', header=None, names=COLUMN_NAMES)
    df_news.head()

    # Generate training dataset statistics
    data_stats = tfdv.generate_statistics_from_dataframe(df_news)
    # Visualize training dataset statistics
    tfdv.visualize_statistics(data_stats)

    # remove unwanted columns and only leave title and category
    df = df_news[['title', 'category']]
    df.head()

    # clean the title
    df['title'] = df['title'].apply(preprocess_text)
    # convert text to a matrix of TF-IDF matrix.
    tfidf_matrix, vectorizer = compute_tfidf(df['title'], stop_words, ngram_range, max_features)

    # encode labels
    labels = df['category'].unique()
    y = encode_labels(labels=labels, data=df['category'])

    # split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, y, test_size=0.2, stratify=y)
    print('Training set Shape: {}  | Test set Shape: {}'.format(X_train.shape, X_test.shape))

    # build and train models
    # Logistic Regression Model
    log_model = LogisticRegression(penalty='l2', max_iter=1000)
    log_model, log_training_time = get_model(log_model, X_train, y_train, LOG_MODEL_NAME)
    log_predictions, log_accuracy, log_metrics_report, log_model_prf = evaluate_mode(log_model, X_test, y_test, labels)
    show_model_performance(log_model, log_accuracy, log_metrics_report, log_training_time,  X_test, y_test, labels)
