""" This module contains class DocumentClassifier."""
import datetime
import pickle
from loguru import logger
import pandas as pd
from typing import List

from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

from document_classifier.core.definitions import LOG_MODEL_NAME, PATH_NEWS, COLUMN_NAMES
from document_classifier.core.functions import preprocess_text
from document_classifier.schema.classification import ModelInput, ModelOutput, ClassificationResponse


class DocumentClassifier:
    """ This class makes prediction by the trained model."""
    def __init__(self, input: ModelInput):
        self.input = input.text
        self.model_path = LOG_MODEL_NAME
        self.model = self.load_model()
        self.data = pd.read_csv(PATH_NEWS, sep='\t', header=None, names=COLUMN_NAMES)
        self.vectorizer = self.build_tfidf_vectorizer()
        self.encoder = self.build_encoder()

    def build_tfidf_vectorizer(self):
        """ Build tfidf vectorizer to convert text to a matrix of TF-IDF features.

        Returns:
            vectorizer: Learned vocabulary and idf from text set
        """
        df = self.data[['title']]
        df['title'] = df['title'].apply(preprocess_text)
        vectorizer = TfidfVectorizer(
            input='content',
            stop_words='english',
            ngram_range=(1, 1),
            min_df=3,
            max_df=0.9,
            max_features=None
        )
        vectorizer.fit(df['title'])
        return vectorizer

    def build_encoder(self):
        """ Encode target labels with value between 0 and n_classes-1."""
        labels = self.data['category'].unique()
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(labels)
        return label_encoder

    def load_model(self):
        """ Load trained model

        Returns:
            model: logistic regression model.
        """
        try:
            model = pickle.load(open(self.model_path, 'rb'))
        except Exception as e:
            logger.exception("Model doesn't exist, please train a new model.", e)
        return model

    def process_result(self, output: List[str]) -> List[ModelOutput]:
        """ Converts prediction results to ModelOutput format.

        Args:
            output: list of output data
        Returns:
            model_output: output in format of ModelOutput

        """
        model_output: List[ModelOutput] = []
        zipped_data = zip(self.input, output)
        for i in zipped_data:
            result = ModelOutput(
                input_text=i[0],
                news_type=i[1]
            )
            model_output.append(result)
        return model_output

    def make_prediction(self) -> ClassificationResponse:
        """ Make predictions for the input.

        Returns:
            response: response from the model in format of ClassificationResponse
        """
        input = self.vectorizer.transform(self.input)
        output = self.model.predict(input)
        output = self.encoder.inverse_transform(output)
        try:
            model_output = self.process_result(output)
            response = ClassificationResponse(
                success=True,
                status='Success!',
                timestamp=datetime.datetime.now(),
                data=model_output
            )
        except Exception as e:
            response = ClassificationResponse(
                success=False,
                status=str(e),
                timestamp=datetime.datetime.now(),
                data=None
            )

        return response


if __name__ == '__main__':
    input = ModelInput(text=[
        "Fed Charles Plosser sees high bar for change in pace of tapering",
        "Health Care Sign-Ups May Miss Goal"
    ])
    classifier = DocumentClassifier(input=input)
    result = classifier.make_prediction()
    print(result)
