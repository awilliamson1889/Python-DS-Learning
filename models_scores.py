import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from src.reader import DSReader
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

dataset_path = os.path.abspath("./tests/datasets/emails.csv")
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def data_set_cleaning(obj):
    obj.to_lower()
    obj.remove_digits()
    obj.remove_punctuation_marks()
    obj.remove_duplicates()
    obj.remove_stopwords()


def load_data():
    df = DSReader(dataset_path)
    data_set_cleaning(df)
    X, y = df.vectorize()
    return X, y


def tokenize(text):
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def display_results(y_test, y_pred):
    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)


X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5)

clf_list = {'RandomForestClassifier1 + tfidf': RandomForestClassifier(random_state=1),
            'RandomForestClassifier2 + chi': RandomForestClassifier(random_state=1),
            'MultinomialNB1': MultinomialNB(),
            'MultinomialNB2 + tfidf': MultinomialNB(),
            'SGDClassifier1 + chi': SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=1),
            'SGDClassifier2 + tfidf': SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=1)}

models_score = pd.DataFrame(columns=['model', 'accuracy', 'precision', 'recall'],
                            index=range(len(clf_list)))

index = 0

for clf in clf_list:

    print(f"step {index + 1} out of {len(clf_list)}")
    if clf == 'MultinomialNB1':
        pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                             ('clf', clf_list[clf])])
    elif clf == 'MultinomialNB2 + tfidf':
        pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                             ('tfidf', TfidfTransformer()),
                             ('clf', clf_list[clf])])
    elif clf == 'SGDClassifier1 + chi':
        pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                             ('chi', SelectKBest(chi2, k=12000)),
                             ('clf', clf_list[clf])])
    elif clf == 'RandomForestClassifier2 + chi':
        pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                             ('chi', SelectKBest(chi2, k=12000)),
                             ('clf', clf_list[clf])])
    else:
        pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                             ('tfidf', TfidfTransformer()),
                             ('clf', clf_list[clf])])

    CVS_accuracy = cross_val_score(pipeline, X.ravel(), y, scoring='accuracy')
    CVS_precision = cross_val_score(pipeline, X.ravel(), y, scoring='precision')
    CVS_recall = cross_val_score(pipeline, X.ravel(), y, scoring='recall')

    # train classifier
    pipeline.fit(X_train.ravel(), y_train)

    # predict on test data
    y_pred = pipeline.predict(X_test.ravel())

    # display results
    display_results(y_test, y_pred)

    models_score.loc[index, 'model'] = str(clf)
    models_score.loc[index, 'accuracy'] = round(CVS_accuracy.mean(), 3)
    models_score.loc[index, 'precision'] = round(CVS_precision.mean(), 3)
    models_score.loc[index, 'recall'] = round(CVS_recall.mean(), 3)

    index = index + 1

models_score = models_score.sort_values(by=['precision', 'recall'], ascending=False)

html_str = models_score.to_html()
html_str = html_str.replace('\n', '')

with open('html_str.txt', 'w') as outfile:
    outfile.write(html_str)