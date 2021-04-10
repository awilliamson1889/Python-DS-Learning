import os.path
from src.reader import DSReader
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


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

print('Fit process successful ending!')

email_text = input('Paste your email text:\n')
while email_text != 'EXIT':
    email_text = DSReader.str_cleaning(email_text)
    predict = pipeline.predict([email_text])
    print("\nI think this is SPAM" if predict == [1] else "I think this is NOT SPAM")
    email_text = input('If you want exit please write: EXIT or write new email text and continue:\n')
