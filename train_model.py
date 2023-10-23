#!/usr/bin/python3

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

df = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

# Split dataset to training and validationing
X_train, X_validation, y_train, y_validation = train_test_split(df['excerpt'], df['target'], test_size=0.2, random_state=42)

# Vectorizing using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_validation_tfidf = vectorizer.transform(X_validation)

# Load data into Linear Regrassion model
regressor = LinearRegression()
regressor.fit(X_train_tfidf, y_train)

# Make a predictions
y_pred = regressor.predict(X_validation_tfidf)

# Evaluate the model using metrics appropriate for regression
mse = mean_squared_error(y_validation, y_pred)
r2 = r2_score(y_validation, y_pred)
print(mse)
print(r2)

# Make predictions on the test data from test.csv
X_test = df_test['excerpt']
X_test_tfidf = vectorizer.transform(X_test)
y_test = regressor.predict(X_test_tfidf)
df_test['target'] = y_test

# Dump result into sumbission.csv
df_test[['id', 'target']].to_csv('submission.csv', index = False)

# Dump trained model and vectorizer
joblib.dump(regressor, 'trained_regression_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
