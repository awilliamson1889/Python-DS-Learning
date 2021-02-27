import os
import unittest
import numpy as np

from src.reader import DSReader

dataset_path = os.path.abspath("tests/datasets/test_dataset_dict.csv")
dataset_empty_path = os.path.abspath("tests/datasets/test_dataset_empty.csv")
dataset_split_path = os.path.abspath("tests/datasets/test_dataset_split.csv")


class TestDatasetReaderNumpy(unittest.TestCase):
    """Checks for numpy related methods of Dataset class"""

    def test_reader__make_dictionary_valid(self):  # ------------------------------------- sorted +++++++
        """Check make_dictionary method"""

        result = [
            "During", "this", "webinar", "we", "will", "cover", "what",
            "is", "DevOps", "and", "Cloud", "Native", "New", "came", "up", "storage"]
        reader = DSReader(dataset_path)
        reader.make_dictionary()

        self.assertEqual(len(set(reader.dictionary)), len(result))
        # self.assertEqual(reader.dictionary, result)
        self.assertEqual(sorted(reader.dictionary), sorted(result))

    def test_reader__make_dictionary_empty(self):
        """Check make_dictionary method on the empty dataset"""

        reader = DSReader(dataset_empty_path)
        reader.make_dictionary()
        self.assertEqual(0, len(reader.dictionary))

    def test_reader__vectorize(self):  # ----------------------------------
        """Check vectorize method"""

        x_result = [['During this webinar we will cover what is DevOps and Cloud Native'],
                    ['New webinar came up'],
                    ['During this webinar we will cover what is DevOps and Cloud Native and storage']]

        # x_result = [
        #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        #     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1]]
        y_result = [1, 1, 0]

        reader = DSReader(dataset_path)
        reader.make_dictionary()
        X, y = reader.vectorize()

        for i, row in enumerate(X):
            self.assertEqual(x_result[i], row.tolist())
        self.assertEqual(y_result, y.tolist())

    def test_reader__vectorize_empty(self):  # ++++++++++++++++++++++++++++++++++
        """Check vectorize method on the empty"""

        reader = DSReader(dataset_empty_path)
        reader.make_dictionary()
        X, y = reader.vectorize()
        self.assertEqual([], X.tolist())
        self.assertEqual([], y.tolist())

    def test_reader__split_test_and_train_data(self):  # +++++++++++++++++++++++++++++++++++++++
        """Check split_test_and_train_data method"""

        reader = DSReader(dataset_split_path)
        reader.make_dictionary()
        X, y = reader.vectorize()
        percent = 0.7
        X_train, y_train, X_test, y_test = reader.split_train_and_test(X, y, percent)
        self.assertEqual(X_train.shape[0], X.shape[0] * percent)
        self.assertEqual(X_test.shape[0], X.shape[0] * round(1 - percent, 2))
        self.assertEqual(y_train.shape[0], y.shape[0] * percent)
        self.assertEqual(y_test.shape[0], y.shape[0] * round(1 - percent, 2))

    def test_reader__split_test_and_train_data_empty(self):  # --------------------------   raise +++++
        """Check split_test_and_train_data method on the empty dataset"""

        reader = DSReader(dataset_empty_path)
        reader.make_dictionary()
        X, y = reader.vectorize()
        percent = 0.7
        with self.assertRaises(Exception):
            X_train, y_train, X_test, y_test = reader.split_train_and_test(X, y, percent)

    def test_reader__split_test_and_train_data_zero_size(self):  # -------------------------------  raise +++++
        """Check split_test_and_train_data method with argument size equals to zero"""

        reader = DSReader(dataset_split_path)
        reader.make_dictionary()
        X, y = reader.vectorize()
        percent = 0
        with self.assertRaises(Exception):
            X_train, y_train, X_test, y_test = reader.split_train_and_test(X, y, percent)