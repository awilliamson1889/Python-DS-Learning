from src.reader import DSReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd
import os

my_data_test = os.path.abspath('../tests/datasets/my_test_dataset.csv')
my_data_test1 = os.path.abspath('../tests/datasets/test_dataset_1_digits.csv')

# my_dataset = DSReader(my_data_test)

# my_dataset.to_lower()
# my_dataset.remove_digits()
# my_dataset.remove_punctuation_marks()
# my_dataset.remove_duplicates()
# my_dataset.remove_stopwords()
# my_dataset.remove_stopwords()

# print(my_dataset.dataset)

my_dataset1 = DSReader('C:/Users/Masquerade/Downloads/emails.csv')

my_dataset1.to_lower()
my_dataset1.remove_digits()
my_dataset1.remove_punctuation_marks()
my_dataset1.remove_duplicates()
my_dataset1.remove_stopwords()
my_dataset1.remove_stopwords()

# print(my_dataset1.dataset)

list_email, list_label = my_dataset1.vectorize()
print(list_email.shape)
print(list_label.shape)

X, y = list_email, list_label
# X, y = my_dataset1.dataset.email, my_dataset1.dataset.label

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values)
print("______________________________________________")
print(y_test)
print("______________________________________________")

vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(X_train.ravel())

print("______________________________________________")
print(counts)
print("______________________________________________")

classifier = MultinomialNB()
targets = y_train
classifier.fit(counts, targets)

examples = ["hello tom, i need your help", "http www Thousands purchased around the world every day","you win won money click heare",
            """Help wanted.  We are a 14 year old fortune 500 company, that is
growing at a tremendous rate.  We are looking for individuals who
want to work from home.This is an opportunity to make an excellent income.  No experience
is required.  We will train you.So if you are looking to be employed from home with a career that has
vast opportunities, then go:http://www.basetel.com/wealthnowWe are looking for energetic and self motivated people.  If that is you
than click on the link and fill out the form, and one of our
employement specialist will contact you.To be removed from our link simple go to""", "hi josh i need money" ,"money money money money money money money money money money money money money money money money money money money money money money money money money money ", """Estate of late Mr.D Dear M, I am a lawyer Harrson Harhog, attorney with the law. I make this offer to you in relation to the death of the late Mr. D.M, who was my client before his death, leaving some huge sum of money (eight million five hundred thousand U.S. dollars) in the bank. """]

example_count = vectorizer.transform(X_test.ravel())
example_count2 = vectorizer.transform(examples)

print("______________________________________________")
print(example_count)
print("______________________________________________")

predictions = classifier.predict(example_count)
predictions2 = classifier.predict(example_count2)
print(predictions)
print(predictions2)
print("Classification rate for NB:", classifier.score(example_count, y_test))

