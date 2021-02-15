from src.reader import DSReader
import os

my_data_test = os.path.abspath('../tests/datasets/my_test_dataset.csv')
my_data_test1 = os.path.abspath('../tests/datasets/test_dataset_1_digits.csv')

my_dataset = DSReader(my_data_test)

my_dataset.to_lower()
my_dataset.remove_digits()
my_dataset.remove_punctuation_marks()
my_dataset.remove_duplicates()
my_dataset.remove_stopwords()
my_dataset.remove_stopwords()

print(my_dataset.dataset)

my_dataset1 = DSReader(my_data_test1)

my_dataset1.to_lower()
my_dataset1.remove_digits()
my_dataset1.remove_punctuation_marks()
my_dataset1.remove_stopwords()
my_dataset1.remove_duplicates()

# 5 steps to clean dataset
# to_lower() -> remove_digits() -> remove_punctuation_marks() -> remove_stopwords() -> remove_duplicates()

print(my_dataset1.dataset)

