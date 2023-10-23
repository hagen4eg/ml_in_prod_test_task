#/usr/bin/python3

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, constr, conlist
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

class UserRequest(BaseModel):
    text: str

class Response(BaseModel):
    score: float

app = FastAPI()

# Load vectorizer and model
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('trained_regression_model.pkl')

@app.post("/get_text_complexity_index", response_model=Response)
def run_model(user_request: UserRequest):
    vectorized_text = vectorizer.transform([user_request.text])
    text_complexity = model.predict(vectorized_text)
    return Response(score=text_complexity[0])
