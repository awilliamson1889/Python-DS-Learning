import nltk
import os.path
import string
import unittest

from src.reader import DSReader


try:
    nltk.find("stopwords")
except LookupError:
    nltk.download("stopwords")

dataset_dup_path = os.path.abspath("tests/datasets/test_dataset_1_dup.csv")
dataset_punctuation_marks = os.path.abspath(
    "tests/datasets/test_dataset_punctuation_marks.csv")
dataset_capital = os.path.abspath("tests/datasets/test_dataset_CAPITAL.csv")
dataset_digits = os.path.abspath("tests/datasets/test_dataset_1_digits.csv")
dataset_stopwords = os.path.abspath("tests/datasets/test_dataset_1_stop_words.csv")


class TestDatasetReaderPandas(unittest.TestCase):

    def test_reader__remove_duplicates(self):
        """Check whether remove_duplicates method removes duplicates
        from dataset based on the email column."""

        NUMBER_DUP = 2
        reader = DSReader(dataset_dup_path)
        before_remove = reader.dataset.shape[0]
        reader.remove_duplicates()

        self.assertEqual(reader.dataset.shape[0], before_remove - NUMBER_DUP)

    def test_reader__to_lower(self):
        """Check whether to_lower method convert all emails in dataset
        to lower case."""

        reader = DSReader(dataset_capital)
        reader.to_lower()
        self.assertEqual(
            True, all(
                [line.email.islower() for line in reader.dataset.itertuples()]
                )
            )

    def test_reader__remove_digits(self):
        """Check the remove_digits method. All emails should not contains
        any digit."""

        reader = DSReader(dataset_digits)
        reader.remove_digits()
        for i, row in reader.dataset.iterrows():
            for word in row['email'].split(' '):
                res = any([
                    digit in str(word) for digit in string.digits
                ])
                self.assertEqual(False, res)

    def test_reader__remove_stopwords(self):
        """Check whether remove_stopwords deletes all 'low weight' words
        from the dataset"""

        reader = DSReader(dataset_stopwords)
        reader.remove_stopwords()
        stop_words = tuple(nltk.corpus.stopwords.words('english'))
        for i, row in reader.dataset.iterrows():
            for word in row['email'].split(' '):
                self.assertEqual(word.lower() not in stop_words, True)

    def test_reader__remove_punctuation_marks(self):
        """Check whether remove_punctuation_marks deletes all punctuation
        marks from the dataset"""

        reader = DSReader(dataset_punctuation_marks)
        reader.to_lower()
        reader.remove_punctuation_marks()
        for i, row in reader.dataset.iterrows():
            for word in row['email'].split(' '):
                # self.assertEqual(word not in string.punctuation, True)
                self.assertEqual(
                    all([mark not in word for mark in string.punctuation]),
                    True
                    )
