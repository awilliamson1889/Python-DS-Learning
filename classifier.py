from src.reader import DSReader
from joblib import load

clf = load('finalized_model.sav')

email_text = input('Paste your email text:\n')
while email_text != 'EXIT':
    email_text = DSReader.str_cleaning(email_text)
    predict = clf.predict([email_text])
    print("\nI think this is SPAM" if predict == [1] else "I think this is NOT SPAM")
    email_text = input('If you want exit please write: EXIT or write new email text and continue:\n')
