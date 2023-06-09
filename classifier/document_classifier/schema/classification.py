""" Define classes for data"""
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime


class ModelInput(BaseModel):
    """ Interface for starting the classification job."""
    text: List[str]


class ModelOutput(BaseModel):
    """ Interface for predictions"""
    input_text: str
    news_type: str


class ClassificationResponse(BaseModel):
    """ Defines prediction status and results"""
    success: bool
    status: str
    timestamp: datetime
    data: Optional[List[ModelOutput]]
