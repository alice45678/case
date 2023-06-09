""" Defines the dependencies used by the Web API."""
from document_classifier.services.classification import ClassificationService


def classification_service():
    return ClassificationService()
