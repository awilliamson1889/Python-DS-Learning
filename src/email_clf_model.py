import os
import pandas as pd
from src.reader import DSReader
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

dataset_path = os.path.abspath("../tests/datasets/emails.csv")
emails_data = DSReader(dataset_path)

emails_data.to_lower()
emails_data.remove_digits()
emails_data.remove_punctuation_marks()
emails_data.remove_duplicates()
emails_data.remove_stopwords()

X, y = emails_data.vectorize()

size = [0.15, 0.25, 0.5, 0.75, 0.95]

for s in size:
    print('TEST SIZE! TEST SIZE! TEST SIZE! = ', s)
    # X_train, y_train, X_test, y_test = emails_data.split_train_and_test(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=s)

    pipeline_MultinomialNB1 = Pipeline([('vect', CountVectorizer()),
                                        ('clf', MultinomialNB())])

    pipeline_MultinomialNB2 = Pipeline([('vect', CountVectorizer()),
                                        ('tfidf', TfidfTransformer()),
                                        ('clf', MultinomialNB())])

    pipeline_MultinomialNB3 = Pipeline([('vect', CountVectorizer()),
                                        ('chi', SelectKBest(chi2, k=1200)),
                                        ('clf', MultinomialNB())])

    pipeline_SGDClassifier1 = Pipeline([('vect', CountVectorizer()),
                                        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                                              alpha=1e-3, random_state=42))])

    pipeline_SGDClassifier2 = Pipeline([('vect', CountVectorizer()),
                                        ('tfidf', TfidfTransformer()),
                                        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                                              alpha=1e-3, random_state=42))])

    pipeline_SGDClassifier3 = Pipeline([('vect', CountVectorizer()),
                                        ('chi', SelectKBest(chi2, k=1200)),
                                        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                                              alpha=1e-3, random_state=42))])

    pipeline_RandomForestClassifier1 = Pipeline([('vect', CountVectorizer()),
                                                 ('clf', RandomForestClassifier())])

    pipeline_RandomForestClassifier2 = Pipeline([('vect', CountVectorizer()),
                                                 ('tfidf', TfidfTransformer()),
                                                 ('clf', RandomForestClassifier())])

    pipelines = [pipeline_MultinomialNB1, pipeline_MultinomialNB2, pipeline_MultinomialNB3,
                 pipeline_SGDClassifier1, pipeline_SGDClassifier2, pipeline_SGDClassifier3,
                 pipeline_RandomForestClassifier1, pipeline_RandomForestClassifier2]

    for pipeline in pipelines:
        pipeline.fit(X_train.ravel(), y_train)

    print("Classification rate for MultinomialNB1:", pipeline_MultinomialNB1.score(X_test.ravel(), y_test))
    print("Classification rate for MultinomialNB2:", pipeline_MultinomialNB2.score(X_test.ravel(), y_test))
    print("Classification rate for MultinomialNB3:", pipeline_MultinomialNB3.score(X_test.ravel(), y_test))
    print("Classification rate for SGDClassifier1:", pipeline_SGDClassifier1.score(X_test.ravel(), y_test))
    print("Classification rate for SGDClassifier2:", pipeline_SGDClassifier2.score(X_test.ravel(), y_test))
    print("Classification rate for SGDClassifier3:", pipeline_SGDClassifier3.score(X_test.ravel(), y_test))
    print("Classification rate for RandomForestClassifier1:", pipeline_RandomForestClassifier1.score(X_test.ravel(), y_test))
    print("Classification rate for RandomForestClassifier2:", pipeline_RandomForestClassifier2.score(X_test.ravel(), y_test))
