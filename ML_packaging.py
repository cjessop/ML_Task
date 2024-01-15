# Import all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import all necessary libraries for machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import mean_squared_error

# Import cross validation libraries
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Import all necessary libraries for deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

from abc import ABC, abstractmethod
import inspect

# Import all necessary libraries for natural language processing
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Import all necessary libraries for time series analysis
# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from pmdarima import auto_arima
# from statsmodels.tsa.arima_model import ARIMA
# from statsmodels.tsa.statespace.sarimax import SARIMAX


# Import all necessary libraries for image processing
from PIL import Image
# from skimage import io
# from skimage.transform import resize
# from skimage.color import rgb2gray
# from skimage.feature import hog
# from skimage import exposure


class BasePredictor(ABC):

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


    # Open source class method to get parameters
    @classmethod
    def _get_param_names(cls):
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            return []
        
        init_signature = inspect.signature(init)
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError('scikit-learn estimators should always '
                                   'specify their parameters in the signature'
                                   ' of their __init__ (no varargs).')
            
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])
    
    #Copied from sklean 
    def get_params(self, deep=True):
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
            
        return out


    def reset(self):
        new = self.__class__(**self.get_params())
        return new

    def load_params(self, params=None):
        self = self.__class__(**params)
        print("params loaded")
        
        return self


# Initialise a class that contains all machinery to format data and apply numerous machine learning algorithms to it
class ML(BasePredictor):
    def __init__(self, data):
        self.data = data

    # Split data into X and y
    def split_X_y(self, X, y):
        X = self.data.drop(y, axis=1)
        y = self.data[y]
        return X, y
    
    # Function to encode categorical data
    def encode_categorical(self, X, y):
        X = pd.get_dummies(X, drop_first=True)
        y = pd.get_dummies(y, drop_first=True)
        return X, y
    
    # Function to deal with missing data
    def missing_data(self, X, y, strategy='mean'):
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
        X = imputer.fit_transform(X)
        y = imputer.fit_transform(y)
        return X, y

    # Function to extract features from classification data
    def extract_features(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test

    # Function to split data into training and testing sets
    def split_data(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test
    
    # Function to scale data
    def scale_data(self, X_train, X_test):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test
    
    # Function to apply logistic regression
    def logistic_regression(self, X_train, X_test, y_train, y_test):
        logmodel = LogisticRegression()
        logmodel.fit(X_train, y_train)
        predictions = logmodel.predict(X_test)
        print(classification_report(y_test, predictions))
        return logmodel
    
    # Function to apply KNN
    def knn(self, X_train, X_test, y_train, y_test, n_neighbors=1):
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)
        print(classification_report(y_test, pred))
        return knn

    # Function to apply SVM
    def svm(self, X_train, X_test, y_train, y_test, kernel='linear'):
        svc_model = SVC()
        svc_model.fit(X_train, y_train, kernel=kernel)
        predictions = svc_model.predict(X_test)
        print(classification_report(y_test, predictions))
        return svc_model
    
    # Function to apply decision tree
    def decision_tree(self, X_train, X_test, y_train, y_test, max_depth=8):
        dtree = DecisionTreeClassifier()
        dtree.fit(X_train, y_train, max_depth=max_depth)
        predictions = dtree.predict(X_test)
        print(classification_report(y_test, predictions))
        return dtree
    
    # Function to apply random forest
    def random_forest(self, X_train, X_test, y_train, y_test, n_estimators=100, max_depth=8):
        rfc = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        rfc.fit(X_train, y_train)
        predictions = rfc.predict(X_test)
        print(classification_report(y_test, predictions))
        return rfc
    
    # Function to apply naive bayes
    def naive_bayes(self, X_train, X_test, y_train, y_test):
        nb = MultinomialNB()
        nb.fit(X_train, y_train)
        predictions = nb.predict(X_test)
        print(classification_report(y_test, predictions))
        return nb
    
    # Function to apply cross validation
    def cross_validation(self, model, X, y, cv=5):
        scores = cross_val_score(model, X, y, cv=cv)
        print(scores)
        print(scores.mean())
        return scores

    # Function to apply grid search
    def grid_search(self, model, param_grid, X, y, cv=5):
        grid = GridSearchCV(model, param_grid, cv=cv)
        grid.fit(X, y)
        print(grid.best_params_)
        print(grid.best_estimator_)
        return grid
    
    # Function to apply randomised search
    def randomised_search(self, model, param_grid, X, y, cv=5, n_iter=100):
        random = RandomizedSearchCV(model, param_grid, cv=cv, n_iter=n_iter)
        random.fit(X, y)
        print(random.best_params_)
        print(random.best_estimator_)
        return random
    
    # Function to apply a multi-layer perceptron
    def mlp(self, X_train, X_test, y_train, y_test, hidden_layers=1, neurons=8, activation='relu', optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], epochs=25, batch_size=32, validation_split=0.2, verbose=1):
        model = Sequential()
        model.add(Dense(neurons, activation=activation))
        for i in range(hidden_layers):
            model.add(Dense(neurons, activation=activation))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
        model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose, callbacks=[early_stop])
        loss_df = pd.DataFrame(model.history.history)
        loss_df.plot()
        predictions = model.predict_classes(X_test)
        print(classification_report(y_test, predictions))
        return model
    
class ffnn(BasePredictor):
    def __init__(self, hidden_layers = [], dropout = 0, epochs = 5, activation = [],batch_size = None):
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.epochs = epochs
        self.activation = activation
        self.batch_size = batch_size

        self.model = Sequential()
        # add the input layer
        for i in range(len(self.hidden_layers)):
            if i == 0:
                self.model.add(Dense(self.hidden_layers[i], activation = self.activation[i], input_dim = self.hidden_layers[i]))
            else:
                self.model.add(Dense(self.hidden_layers[i], activation = self.activation[i]))
            if self.dropout > 0:
                self.model.add(Dropout(self.dropout))

        # add the output layer
        self.model.add(Dense(1, activation = 'sigmoid'))
        # set the loss function and optimiser
        self.model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    def fit(self, X, y):
        try: 
            self.model.fit(X, y, epochs = self.epochs, batch_size = self.batch_size)
        except:
            print('Error in fitting the model')
            pass

    def predict(self, X):
        try:
            return self.model.predict(X)
        except:
            print('Error in predicting the model')
            pass


# Generate some data to test the class using numpy and pandas
data = np.random.randint(0, 100, (1000, 50))
data = pd.DataFrame(data)
data['target'] = np.random.randint(0, 2, 1000)
print(data.head())

# Initialise the class
ml = ML(data)

# Split the data into X and y
X, y = ml.split_X_y(X='target', y='target')
print(X.head())
print(y.head())

# Encode the categorical data
X, y = ml.encode_categorical(X, y)
print(X.head())
print(y.head())

# Deal with missing data
# X, y = ml.missing_data(X, y)
# print(X.head())
# print(y.head())

# Extract features from classification data
# X_train, X_test, y_train, y_test = ml.extract_features(X, y)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = ml.split_data(X, y)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Scale the data
X_train, X_test = ml.scale_data(X_train, X_test)
print(X_train.shape)
print(X_test.shape)

# Apply logistic regression
logmodel = ml.logistic_regression(X_train, X_test, y_train, y_test)
print(logmodel)

# Create meta class to apply all machine learning algorithms
class ML_meta:
    def __init__(self, data):
        self.data = data

    # Call the ML class to apply all machine learning algorithms
    def call_ML(self):
        ml = ML(self.data)
        return ml
    
    # Call the deep learning class to apply all deep learning algorithms
    # def call_DL(self):
    #     dl = DL(self.data)
    

    
