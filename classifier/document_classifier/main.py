""" Module contains the Web API."""
import sys
sys.path.append('/Users/dehongyuan/anaconda3/envs/myenv/lib/python3.8/site-packages')
sys.path.append('/Users/dehongyuan/Desktop/MyProject/Ing_case/Ing')
import uvicorn
from fastapi import FastAPI

from document_classifier.api.routers.classification import router

app = FastAPI()
app.include_router(router, prefix='/api')


if __name__ == '__main__':
    uvicorn.run('main:app')
