import unittest

import numpy as np
from numpy.random import shuffle

from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError

from pantaray.importance_estimation import PCIF


class ImportanceTest(unittest.TestCase):

    def setUp(self):
        data, target = load_digits(return_X_y=True)
        index_0to4 = np.where(target <= 4)[0]
        index_5to9 = np.where(target > 4)[0]
        shuffle(index_0to4)
        shuffle(index_5to9)
        # 75% of 0-4 and 25% of 5-9 in the train set.
        # 25% of 0-4 and 75% of 5-9 in the test set.
        first, second = int(len(index_0to4) * 0.75), int(len(index_5to9) * 0.25)
        train = np.concatenate((index_0to4[:first], index_5to9[:second]))
        test = np.concatenate((index_0to4[first:], index_5to9[second:]))
        # Put results as instance attributes.
        self.X_train, self.y_train = data[train], data[train]
        self.X_test, self.y_test = data[test], data[test]

    def test_something(self):
        X_train, X_test = self.X_train, self.X_test
        pcif = PCIF()
        pcif.fit(X_train, X_test, LogisticRegression())
        w_train = pcif.predict(X_train)
        w_test = pcif.predict(X_test)
        print(w_train.mean())
        print(w_test.mean())

    def test_error_predict_without_fit(self):
        X_train, X_test = self.X_train, self.X_test
        pcif = PCIF()
        with self.assertRaises(NotFittedError):
            w_train = pcif.predict(X_train)
