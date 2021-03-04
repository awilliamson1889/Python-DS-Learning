import pandas as pd
from src.reader import DSReader
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


emails_data = DSReader('C:/Users/Masquerade/Downloads/emails.csv')

emails_data.to_lower()
emails_data.remove_digits()
emails_data.remove_punctuation_marks()
emails_data.remove_duplicates()
emails_data.remove_stopwords()

X, y = emails_data.vectorize()

# X_train, y_train, X_test, y_test = emails_data.split_train_and_test(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y)

pipeline_MultinomialNB = Pipeline([('vect', CountVectorizer()),
                                   ('tfidf', TfidfTransformer()),
                                   ('clf', MultinomialNB()),])

pipeline_SGDClassifier = Pipeline([('vect', CountVectorizer()),
                                   ('tfidf', TfidfTransformer()),
                                   ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                                         alpha=1e-3, random_state=42)),])

pipeline_MultinomialNB.fit(X_train.ravel(), y_train)
pipeline_SGDClassifier.fit(X_train.ravel(), y_train)

predicted_MultinomialNB = pipeline_MultinomialNB.predict(X_test.ravel())
predicted_SGDClassifier = pipeline_SGDClassifier.predict(X_test.ravel())

print(predicted_MultinomialNB)
print(predicted_SGDClassifier)

print("Classification rate for MultinomialNB:", pipeline_MultinomialNB.score(X_test.ravel(), y_test))
print("Classification rate for SGDClassifier:", pipeline_SGDClassifier.score(X_test.ravel(), y_test))

my_test_data = [["Hello you win 4000$, please click here !!! --->"],
                ["Hi Mark. Can you help me please!!!!"],
                ["hy  hy h@!#12ew #@!123k !!!3 kweoq ck ---wjo 634 oko qd"]]

my_test_data_clean = [["hello you win please click here"],
                      ["mark can you help please"],
                      ["hew kweoq wjo oko"]]


for email in my_test_data:
    predicted_test_MultinomialNB = pipeline_MultinomialNB.predict(email)
    predicted_test_SGDClassifier = pipeline_SGDClassifier.predict(email)

    print(email)
    print("predicted MultinomialNB: ", predicted_test_MultinomialNB)
    print("predicted SGDClassifier: ", predicted_test_SGDClassifier)

for email in my_test_data_clean:
    predicted_test_MultinomialNB = pipeline_MultinomialNB.predict(email)
    predicted_test_SGDClassifier = pipeline_SGDClassifier.predict(email)

    print(email)
    print("predicted MultinomialNB: ", predicted_test_MultinomialNB)
    print("predicted SGDClassifier: ", predicted_test_SGDClassifier)
    