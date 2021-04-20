import os.path
from src.reader import DSReader
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from joblib import dump


url = "raw.githubusercontent.com/awilliamson1889/Python-DS-Learning/email-clf-model/tests/datasets/emails.csv"
dataset_path = os.path.abspath("tests/datasets/emails.csv")

print("Please wait model fitting.")

try:
    emails_data = DSReader(dataset_path)
except FileNotFoundError:
    print("Dataset not found.\nDownloading dataset from: ", url)
    emails_data = DSReader(url)

DSReader.dataset_cleaning(emails_data)

X, y = emails_data.vectorize()

pipeline = Pipeline([('vect', CountVectorizer(tokenizer=DSReader.tokenize)),
                     ('clf', MultinomialNB())])

pipeline.fit(X.ravel(), y)

filename = 'finalized_model.sav'
dump(pipeline, filename)

print('Fit process successful ending!')

