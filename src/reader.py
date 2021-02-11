import pandas as pd
import numpy as np
import string
import math
import nltk


class DSReader(object):

    def __init__(self, patch):
        self.patch = patch
        self.dataset = DSReader.load_dataset(self.patch)
        self.dictionary = []

    @staticmethod
    def load_dataset(path):
        data_frame_csv = pd.read_csv(path)
        return data_frame_csv

    def remove_digits(self):

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
        self.dataset['email'] = self.dataset['email'].str.lower()
        return self.dataset

    def remove_punctuation_marks(self):

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
        self.dataset = pd.DataFrame.drop_duplicates(self.dataset)
        return self.dataset

    def make_dictionary(self):
        email_index = self.dataset.index

        for i in email_index:
            inp_str = str(self.dataset['email'][i])
            email_text = inp_str.split()
            self.dictionary.extend(email_text)  # mb append

        return self.dictionary

    def vectorize(self):
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
        train_count = math.floor(list_email.size * 0.75)

        train_labels = np.array(list_label[:train_count])
        test_labels = np.array(list_label[train_count:])

        train_emails = np.array(list_email[:train_count])
        test_emails = np.array(list_email[train_count:])

        return train_emails, train_labels, test_emails, test_labels
