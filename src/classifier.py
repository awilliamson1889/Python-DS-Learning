"""Email classifier fit model"""
import sys
import os
import logging
from joblib import load
from reader import DSReader


sys.path.append('src')

clf = load(os.path.abspath('models/MultinomialNB_finalized_model.sav'))

email_text = input('Paste your email text:\n')
while email_text != 'EXIT':
    logging.debug('Email text:\n%s', email_text)

    email_text = DSReader.str_cleaning(email_text)

    logging.debug('Email text clean:\n%s', email_text)
    predict = clf.predict([email_text])
    logging.debug('Predicted label:\n%s', predict)

    print("\nI think this is SPAM" if predict == [1] else "I think this is NOT SPAM")
    email_text = input('If you want exit please write: '
                       'EXIT or write new email text and continue:\n')
