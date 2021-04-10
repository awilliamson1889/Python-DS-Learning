import nltk
import os
import numpy as np
import pandas as pd

from src.reader import DSReader
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

try:
    nltk.find("wordnet")
except LookupError:
    nltk.download("wordnet")

try:
    nltk.find("stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.find("punkt")
except LookupError:
    nltk.download('punkt')

dataset_path = os.path.abspath("tests/datasets/emails.csv")


def load_data():
    df = DSReader(dataset_path)
    DSReader.dataset_cleaning(df)
    X, y = df.vectorize()
    return X, y


def display_results(y_test, y_pred):
    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)


X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5)

clf_list = {'RandomForestClassifier1 + tfidf': RandomForestClassifier(random_state=1),
            'RandomForestClassifier1 + tfidf - tokenizer': RandomForestClassifier(random_state=1),
            'RandomForestClassifier2 + chi': RandomForestClassifier(random_state=1),
            'MultinomialNB1': MultinomialNB(),
            'MultinomialNB2 + tfidf': MultinomialNB(),
            'MultinomialNB3 - tokenizer': MultinomialNB(),
            'MultinomialNB4 + tfidf - tokenizer': MultinomialNB(),
            'SGDClassifier1 + chi': SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=1),
            'SGDClassifier2 + tfidf': SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=1)}

models_score = pd.DataFrame(columns=['model', 'accuracy', 'precision', 'recall', 'F1-score'],
                            index=range(1, len(clf_list)+1))

index = 1
for clf in clf_list:
    print(f"step {index} out of {len(clf_list)}")
    if clf == 'MultinomialNB1':
        pipeline = Pipeline([('vect', CountVectorizer(tokenizer=DSReader.tokenize)),
                             ('clf', clf_list[clf])])
    elif clf == 'MultinomialNB2 + tfidf':
        pipeline = Pipeline([('vect', CountVectorizer(tokenizer=DSReader.tokenize)),
                             ('tfidf', TfidfTransformer()),
                             ('clf', clf_list[clf])])
    elif clf == 'MultinomialNB3 - tokenizer':
        pipeline = Pipeline([('vect', CountVectorizer()),
                             ('clf', clf_list[clf])])
    elif clf == 'MultinomialNB4 + tfidf - tokenizer':
        pipeline = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf', clf_list[clf])])
    elif clf == 'SGDClassifier1 + chi':
        pipeline = Pipeline([('vect', CountVectorizer(tokenizer=DSReader.tokenize)),
                             ('chi', SelectKBest(chi2, k=12000)),
                             ('clf', clf_list[clf])])
    elif clf == 'RandomForestClassifier2 + chi':
        pipeline = Pipeline([('vect', CountVectorizer(tokenizer=DSReader.tokenize)),
                             ('chi', SelectKBest(chi2, k=12000)),
                             ('clf', clf_list[clf])])
    elif clf == 'RandomForestClassifier1 + tfidf - tokenizer':
        pipeline = Pipeline([('vect', CountVectorizer()),
                             ('chi', SelectKBest(chi2, k=12000)),
                             ('clf', clf_list[clf])])
    else:
        pipeline = Pipeline([('vect', CountVectorizer(tokenizer=DSReader.tokenize)),
                             ('tfidf', TfidfTransformer()),
                             ('clf', clf_list[clf])])

    CVS_accuracy = cross_val_score(pipeline, X.ravel(), y, scoring='accuracy')
    CVS_precision = cross_val_score(pipeline, X.ravel(), y, scoring='precision')
    CVS_recall = cross_val_score(pipeline, X.ravel(), y, scoring='recall')

    accuracy = round(CVS_accuracy.mean(), 3)
    precision = round(CVS_precision.mean(), 3)
    recall = round(CVS_recall.mean(), 3)
    f1_score = round(2 * (precision * recall) / (precision + recall), 3)

    # train classifier
    pipeline.fit(X_train.ravel(), y_train)

    # predict on test data
    y_pred = pipeline.predict(X_test.ravel())

    # display results
    display_results(y_test, y_pred)

    models_score.loc[index, 'model'] = str(clf)
    models_score.loc[index, 'accuracy'] = accuracy
    models_score.loc[index, 'precision'] = precision
    models_score.loc[index, 'recall'] = recall
    models_score.loc[index, 'F1-score'] = f1_score

    index = index + 1

models_score = models_score.sort_values(by=['F1-score', 'precision'], ascending=False)

html_str = (models_score.to_html()).replace('\n', '')

with open('scores_table_html.txt', 'w') as outfile:
    outfile.write(html_str)
