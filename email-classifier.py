import os
import pandas as pd
from src.reader import DSReader
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


def data_set_cleaning(obj):
    obj.to_lower()
    obj.remove_digits()
    obj.remove_punctuation_marks()
    obj.remove_duplicates()
    obj.remove_stopwords()

url = "https://raw.githubusercontent.com/awilliamson1889/Python-DS-Learning/email-clf-model/tests/datasets/emails.csv"

try:
    dataset_path = os.path.abspath("./tests/datasets/emails.csv")
    print(dataset_path)
    emails_data = DSReader(dataset_path)
except FileNotFoundError:
    print("Dataset not found.\nDownloading dataset from: ", url)
    emails_data = DSReader(url)

data_set_cleaning(emails_data)

X, y = emails_data.vectorize()

pipeline_MultinomialNB = Pipeline([('vect', CountVectorizer()),
                                   ('clf', MultinomialNB())])

pipeline_MultinomialNB.fit(X.ravel(), y)

print('Fit process successful ending!')

email_text = input('Paste your email text:\n')

df_email = pd.DataFrame({'email': [email_text],
                         'label': '?'})

df_email.to_csv(path_or_buf='user.csv', index=False)

dataset_path = os.path.abspath("user.csv")
emails_user = DSReader(dataset_path)

data_set_cleaning(emails_user)

predict = pipeline_MultinomialNB.predict(emails_user.dataset['email'])

print("\nI think this is SPAM" if predict == [1] else "I think this is NOT SPAM")
input()
