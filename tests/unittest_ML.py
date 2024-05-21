import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from HeimdallX.ML_packaging import ML_meta, ML_post_process

class TestMLMeta(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
        self.data = pd.DataFrame(np.column_stack((self.X, self.y)), columns=list(range(10)) + ['target'])

    def test_split_data(self):
        meta = ML_meta(self.data)
        X, y = meta.split_data(encode_categorical=False)
        self.assertEqual(X.shape, (1000, 10))
        self.assertEqual(y.shape, (1000,))

    def test_apply_all_models(self):
        meta = ML_meta(self.data)
        rf, svm, knn, lr, nb, dt, ec, gbc, abc, scores = meta.apply_all_models(flag=True)
        self.assertIsNotNone(rf)
        self.assertIsNotNone(svm)
        self.assertIsNotNone(knn)
        self.assertIsNotNone(lr)
        self.assertIsNotNone(nb)
        self.assertIsNotNone(dt)
        self.assertIsNotNone(ec)
        self.assertIsNotNone(gbc)
        self.assertIsNotNone(abc)
        self.assertIsNotNone(scores)

    def test_apply_single_model(self):
        meta = ML_meta(self.data, model="SVM")
        meta.apply_single_model()
        # Add more assertions if needed

class TestMLPostProcess(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
        self.data = pd.DataFrame(np.column_stack((self.X, self.y)), columns=list(range(10)) + ['target'])

    def test_split_data(self):
        post_process = ML_post_process(self.data, target='target')
        X, y = post_process.split_data(encode_categorical=False)
        self.assertEqual(X.shape, (1000, 10))
        self.assertEqual(y.shape, (1000,))

    def test_get_X_test(self):
        post_process = ML_post_process(self.data, target='target')
        X_test = post_process.get_X_test()
        self.assertEqual(X_test.shape, (200, 10))

    def test_data_info(self):
        post_process = ML_post_process(self.data, target='target')
        post_process.data_info()
        # This test doesn't assert anything, it just verifies that the method runs without errors

if __name__ == '__main__':
    unittest.main()