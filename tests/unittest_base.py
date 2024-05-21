import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from HeimdallX.BaseMLClasses import ML, svm, Simple_CNN, CNN, ffnn

class TestML(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
        self.data = pd.DataFrame(np.column_stack((self.X, self.y)), columns=list(range(10)) + ['target'])

    def test_split_X_y(self):
        ml = ML(self.data)
        X, y = ml.split_X_y('target')
        self.assertEqual(X.shape, (1000, 10))
        self.assertEqual(y.shape, (1000,))

    def test_encode_categorical(self):
        ml = ML(self.data)
        X, y = ml.encode_categorical(self.X, self.y)
        self.assertEqual(X.shape, (1000, 10))
        self.assertEqual(y.shape, (1000, 2))

    def test_missing_data(self):
        self.data.iloc[0, 0] = np.nan
        ml = ML(self.data)
        X, y = ml.missing_data(self.X, self.y, strategy='mean')
        self.assertFalse(np.isnan(X).any())
        self.assertFalse(np.isnan(y).any())

    def test_extract_features(self):
        ml = ML(self.data)
        X_train, X_test, y_train, y_test = ml.extract_features(self.X, self.y)
        self.assertEqual(len(X_train) + len(X_test), len(self.X))
        self.assertEqual(len(y_train) + len(y_test), len(self.y))

    def test_split_data(self):
        ml = ML(self.data)
        X_train, X_test, y_train, y_test = ml.split_data(self.X, self.y)
        self.assertEqual(len(X_train) + len(X_test), len(self.X))
        self.assertEqual(len(y_train) + len(y_test), len(self.y))

    def test_scale_data(self):
        ml = ML(self.data)
        X_train, X_test, y_train, y_test = ml.split_data(self.X, self.y)
        X_train_scaled, X_test_scaled = ml.scale_data(X_train, X_test)
        self.assertAlmostEqual(X_train_scaled.mean(), 0, places=2)
        self.assertAlmostEqual(X_train_scaled.std(), 1, places=2)
        self.assertAlmostEqual(X_test_scaled.mean(), 0, places=2)
        self.assertAlmostEqual(X_test_scaled.std(), 1, places=2)

class TestSVM(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def test_svm(self):
        svm_model = svm()
        svm_model.run(self.X_train, self.X_test, self.y_train, self.y_test)
        self.assertIsNotNone(svm_model)

class TestFFNN(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def test_ffnn(self):
        ffnn_model = ffnn(hidden_layers=[8, 4], dropout=0.2, epochs=10, activation=['relu', 'relu'], batch_size=32)
        ffnn_model.fit(self.X_train, self.y_train)
        predictions = ffnn_model.predict(self.X_test)
        self.assertEqual(predictions.shape, (len(self.X_test), 1))

if __name__ == '__main__':
    unittest.main()