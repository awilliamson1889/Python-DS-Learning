"""Email classifier"""
import os.path
import sys
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from reader import DSReader


sys.path.append('src')

print(os.path.abspath(""))
dataset_path = os.path.abspath("tests/datasets/emails.csv")

try:
    emails_data = DSReader(dataset_path)
except FileNotFoundError:
    print("Dataset not found.")

print("Please wait model fitting.")

DSReader.dataset_cleaning(emails_data)

X, y = emails_data.vectorize()

pipeline = Pipeline([('vect', CountVectorizer(tokenizer=DSReader.tokenize)),
                     ('clf', MultinomialNB())])

pipeline.fit(X.ravel(), y)

file_name = 'finalized_model.sav'
dump(pipeline, file_name)

print('Fit process successful ending!')
