"""This module does to work with dataset"""
import string
import math
import pandas as pd
import numpy as np
import nltk


class DSReader:
    """The class DSReader is used to clean up the dataset and break the dataset down
    into test and training data. The main use is to clean up the dataset from "trash"
    Attributes
    ----------
    patch : str
        full path to the dataset
    dataset : DataFrame
        dataset
    dictionary : list[str]
        list of all words in the dataset
    Methods
    -------
    load_dataset(path)
        reads a file with a set of data
    remove_digits()
        removes all digits from the data set
    to_lower()
        returns a dataset in lowercase
    remove_punctuation_marks()
        returns a data set without punctuation marks
    remove_stopwords()
        returns a data set without stopwords
    remove_duplicates()
        returns a set of data without duplicate data
    make_dictionary()
        returns a list consisting of all the words in the dataset
    vectorize()
        returns two lists: a list of email texts and a list of labels
    split_train_and_test(list_email: list[str], list_label: list[int])
        return tuple of train and test data
    """

    def __init__(self, patch):
        self.patch = patch
        self.dataset = DSReader.load_dataset(self.patch)
        self.dictionary = []

    @staticmethod
    def load_dataset(path):
        """Method for loading a dataset
        Parameters
        ----------
        path : str
            full path to the dataset
        """

        data_frame_csv = pd.read_csv(path)
        return data_frame_csv

    def remove_digits(self):
        """Method for remove digits from a dataset
        """

        def no_digits(inp_str):
            inp_str = str(inp_str)
            no_digits_str = ''
            for char in inp_str:
                if char not in string.digits:
                    no_digits_str = no_digits_str + char
                else:
                    no_digits_str = no_digits_str + ' '
            return no_digits_str

        self.dataset['email'] = self.dataset['email'].map(no_digits)
        return self.dataset

    def to_lower(self):
        """Method for converting a dataset to lowercase
        """

        self.dataset['email'] = self.dataset['email'].str.lower()
        return self.dataset

    def remove_punctuation_marks(self):
        """Method for remove punctuation marks from a dataset
        """

        def no_punctuation(inp_str):
            inp_str = str(inp_str)
            no_punctuation_str = ""
            for char in inp_str:
                if char not in string.punctuation:
                    no_punctuation_str = no_punctuation_str + char
                else:
                    no_punctuation_str = no_punctuation_str + ' '
            return no_punctuation_str

        self.dataset['email'] = self.dataset['email'].map(no_punctuation)
        return self.dataset

    def remove_stopwords(self):
        """Method for remove stopwords from a dataset
        """

        stop_words = tuple(nltk.corpus.stopwords.words('english'))

        def no_stopwords(inp_str):
            words = nltk.word_tokenize(inp_str)
            new_list_str = ' '.join(words)
            words = (new_list_str.lower()).split()

            without_stop_words = [word.lower() for word in words
                                  if word not in stop_words]
            without_short_words = [x for x in without_stop_words
                                   if len(x) > 2]
            new_str = ' '.join(without_short_words)
            return new_str

        self.dataset['email'] = self.dataset['email'].map(no_stopwords)
        return self.dataset

    def remove_duplicates(self):
        """Method for remove duplicates from a dataset
        """

        self.dataset = pd.DataFrame.drop_duplicates(self.dataset)
        return self.dataset

    def make_dictionary(self):
        """Method for creating a list of all words in a dataset
        """

        email_index = self.dataset.index

        for i in email_index:
            inp_str = str(self.dataset['email'][i])
            email_text = inp_str.split()
            self.dictionary.extend(email_text)  # mb append

        return self.dictionary

    def vectorize(self):
        """Method for creating lists of emails and labels
        """

        email_index = self.dataset.index
        email_list = []

        for i in email_index:
            inp_str = str(self.dataset['email'][i])
            email_text = inp_str.split()
            email_list.append(email_text)

        emails = np.array(email_list, dtype=object)
        emails_labels = np.array(self.dataset['label'])

        return emails, emails_labels

    @staticmethod
    def split_train_and_test(list_email, list_label):
        """Method for splitting data into training and test data
        Parameters
        ----------
        list_email: list[str]
            a list consisting of email texts
        list_label: list[int]
            a list consisting of email labels
        """

        train_count = math.floor(list_email.size * 0.75)

        train_labels = np.array(list_label[:train_count])
        test_labels = np.array(list_label[train_count:])

        train_emails = np.array(list_email[:train_count])
        test_emails = np.array(list_email[train_count:])

        return train_emails, train_labels, test_emails, test_labels
