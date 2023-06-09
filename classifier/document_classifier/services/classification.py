""" This module contains classification services."""
import datetime
from loguru import logger

from document_classifier.core.classifier import DocumentClassifier
from document_classifier.schema.classification import ModelInput, ClassificationResponse


class ClassificationService:
    """ Define service for text classification."""
    @staticmethod
    def run_model(request: ModelInput) -> ClassificationResponse:
        """ Call make_prediction function of DocumentClassifier.

        Args:
            request: input in format of ModelInput

        Returns:
            result: output of the classifier
        """
        document_classifier = DocumentClassifier(request)
        logger.info("Starting prediction...")

        try:
            result = document_classifier.make_prediction()

        except Exception as e:
            logger.exception("Unknown error during prediction: ", e)
            result = ClassificationResponse(
                success=False,
                status=str(e),
                timestamp=datetime.datetime.now(),
                data=None
            )
        return result
