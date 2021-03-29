import os
import nltk
import matplotlib.pyplot as plt
from src.reader import DSReader
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

nltk.download('punkt')

try:
    nltk.find("stopwords")
except LookupError:
    nltk.download("stopwords")


def data_set_cleaning(obj):
    obj.to_lower()
    obj.remove_digits()
    obj.remove_punctuation_marks()
    obj.remove_duplicates()
    obj.remove_stopwords()

url = "raw.githubusercontent.com/awilliamson1889/Python-DS-Learning/email-clf-model/tests/datasets/emails.csv"
dataset_path = os.path.abspath("./tests/datasets/emails.csv")
test_dataset_path = os.path.abspath("./tests/datasets/spam_ham_dataset.csv")

test_data = DSReader(test_dataset_path)

try:
    emails_data = DSReader(dataset_path)
except FileNotFoundError:
    print("Dataset not found.\nDownloading dataset from: ", url)
    emails_data = DSReader(url)

data_set_cleaning(emails_data)
data_set_cleaning(test_data)

X_train, y_train = emails_data.vectorize()
X_test, y_test = test_data.vectorize()

pipeline_SGDClassifier2 = Pipeline([('vect', CountVectorizer()),
                                    ('tfidf', TfidfTransformer()),
                                    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                                          alpha=1e-3, random_state=1))])

pipeline_SGDClassifier2.fit(X_train.ravel(), y_train)

print('Fit process successful ending!')

answers = pipeline_SGDClassifier2.predict(X_test.ravel())

print(pipeline_SGDClassifier2.score(X_test.ravel(), y_test))


class_names = ['not spam', 'spam']

titles_options = [("Confusion matrix pipeline_SGDClassifier", None, pipeline_SGDClassifier2)]

for title, normalize, classifier in titles_options:
    disp = plot_confusion_matrix(classifier, X_test.ravel(), y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)

    matrix = disp.confusion_matrix

    TP, FP, FN, TN = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]

    disp.ax_.set_title(title)

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f_measure = 2 * recall * precision / (precision + recall)

    plt.show()

with open('metrics.txt', 'w') as outfile:
    outfile.write(f'Matrix:\n{matrix}\nRecall = {recall}\nPrecision = {precision}\nF-measure = {f_measure}')

