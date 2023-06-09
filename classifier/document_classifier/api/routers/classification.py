""" Prediction Web API"""
from fastapi import APIRouter, Depends

from document_classifier.api import dependencies
from document_classifier.schema.classification import ModelInput, ClassificationResponse
from document_classifier.services.classification import ClassificationService

router = APIRouter()


@router.post('/prediction', response_model=ClassificationResponse, tags=['Document Classification'])
def predict(
        request: ModelInput,
        classification_service: ClassificationService = Depends(dependencies.ClassificationService)
):
    response = classification_service.run_model(request=request)
    return response
