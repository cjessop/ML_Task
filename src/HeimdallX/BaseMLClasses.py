# Import all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import builtins
import yaml
import re
from Config import config
from _items import Ref, Call, Item
from importlib import import_module
import pickle

# Import all necessary libraries for machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier

# Import cross validation libraries
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Import all necessary libraries for deep learning
try:
    import tensorflow as tf
    from keras import layers
    #from keras.utils import get_cust_objects
    from keras.models import Sequential
    from keras.layers import Input
    from keras.layers import Dense, Dropout, Lambda
    #from tf.keras.preprocessing.image import ImageDataGenerator
    from keras.utils import image_dataset_from_directory    
    from keras.callbacks import EarlyStopping
    from keras.models import load_model
    from keras.applications import MobileNetV2
except ImportError:
    print("Unable to Import Tensorflow/Keras inside of the Base Classes script")
    pass

from abc import ABC, abstractmethod
import inspect


model_dict = {
    "SupportVector": "SVM",
    "KNearestNeighbour": "kNN",
    "LinearRegression": "LinReg",
    "NaiveBayes": "NB",
    "MultiLayerPerceptron": "MLP",
    "DecisionTree": "DT",
    "RandomForest": "RF",
    "NeuralNetwork": "NN"
}


class BasePredictor(ABC):
    """
    Base class for predictive models.

    Defines the core functionalities expected from predictive models, such as fitting and predicting.
    Also provides utility methods for parameter management and model resetting/loading.

    Attributes:
        _get_param_names (classmethod): Retrieves the names of parameters for the class's constructor.
        get_params (method): Returns a dictionary of the class's hyperparameters.
        reset (method): Creates a new instance of the class with the same parameters as the current instance.
        load_params (method): Loads new parameters into the class instance.

    Abstract Methods:
        fit (abstractmethod): Trains the model on given data.
        predict (abstractmethod): Makes predictions on new data using the trained model.
    """

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


    # Open source class method to get parameters
    @classmethod # Decorator to make the method accessible without creating a class instance
    def _get_param_names(cls):
        """
        Retrieves the names of parameters for the class's constructor.

        Returns:
            list: A sorted list of parameter names, excluding 'self'.
        """
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            return [] # If using the default object constructor, return an empty list
        
        init_signature = inspect.signature(init) # Get the constructor's signature
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD] # Exclude 'self' and variable-length arguments
        
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError('scikit-learn estimators should always '
                                   'specify their parameters in the signature'
                                   ' of their __init__ (no varargs).')
            
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])
    
    #Copied from sklean 
    def get_params(self, deep=True):
        """
        Returns a dictionary of the class's hyperparameters.

        Args:
            deep (bool, optional): If True, includes parameters for sub-objects that have a get_params() method. Defaults to True.

        Returns:
            dict: A dictionary of parameter names mapped to their values.
        """
        out = dict() # Initialise the output dictionary
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, 'get_params'): # If deep is True and the value has a get_params() method:
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items) 
            out[key] = value # Add the parameter to the output dictionary
             
        return out # Return the dictionary of parameters


    def reset(self):
        new = self.__class__(**self.get_params())
        return new

    def load_params(self, params=None):
        self = self.__class__(**params)
        print("params loaded")
        
        return self
    
def import_function(qualified_name):
    path = qualified_name.split('.')
    module = builtins

    for i, key in enumerate(path[:,1]):
        if not hasattr(module, key):

            module = import_module('.'.join(path[:i+1]))
        else:
            module = getattr(module, key)
        
    function = getattr(module, path[-1])
    
    return function

class PipelineLoader(yaml.SafeLoader):
    @classmethod
    def load(cls, instream):
        loader = cls(instream)

        try:
            return loader.get_single_data()
        finally:
            loader.dispose()

    def construct_map(self, node, deep=False):
        mapping = super().construct_mapping(node, deep)

        for key in mapping:
            if not isinstance(key, str):
                raise ValueError(f'key {key} is not the correct type (string), it is {type(key).__name__} \n'
                                 f'{node.start_mark}')
            
        return mapping
    
    def construct_ref(self, node):
        ref = self.construct_scalar(node)
        
        if ref[:1] == "$":
            ref = ref[1:]
        if not ref:
            raise ValueError(f'An empty reference was provided {node.start_mark}')
        
        return ref(ref)
    
    def construct_call(self, name, node):

        try: 
            object = import_function(name)
        except (ModuleNotFoundError, AttributeError) as err:
            raise ImportError(f'{err} ') from err
        
        if isinstance(node, yaml.ScalarMode):
            if node.value:
                raise ValueError(f'The ScalarNode {node.value} has to be empty to import the object')
            return object
        
        else:
            if isinstance(node, yaml.SequenceNode):
                args = self.construct_sequence(node)
                kwargs = {}
            elif isinstance(node, yaml.MappingNode):
                args = []
                kwargs = self.construct_map(node)

            return Call(object, args, kwargs)
        
    # def construct_quant(self, node):
    #     val = self.construct_scalar(node)

    #     return quantity(val)

    #PipelineLoader.add_constructor('!ref', PipelineLoader.construct_ref)
    #PipelineLoader.add_implicit_resolver('!ref', re.compile(r'\$\w+'), ['$'])
    #PipelineLoader.add_multi_resolver('!', PipelineLoader.construct_call)
    #PipelineLoader.add_constructor('!quantity', PipelineLoader.construct_quant)
    #PipelineLoader.add_implicit_resolver()


    def load_pipeline_yaml(filename):
        with open(filename, 'r') as data:
            return PipelineLoader.load(data) or {}
 
        



# Initialise a class that contains all machinery to format data and apply numerous machine learning algorithms to it
class ML(BasePredictor):
    """
    Initializes the ML class with the given data.

    Args:
        data (pd.DataFrame): The dataset to work with.
    """
    def __init__(self, data):
        self.data = data


    # Split data into X and y
    def split_X_y(self, y):
        """
        Splits the data into features (X) and target variable (y).

        Args:
            y (str): Name of the target variable column.

        Returns:
            tuple: (X, y) where X is the feature matrix and y is the target vector.
        """
        X = self.data.drop(y, axis=1) # Drop the target column to get features
        y = self.data[y] # Extract the target column
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
    def lr(self, X_train, X_test, y_train, y_test):
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
    def svm(self, X_train, X_test, y_train, y_test, kernel='rbf'):
        svc_model = SVC(kernel=kernel)
        svc_model.fit(X_train, y_train)
        predictions = svc_model.predict(X_test)
        print(classification_report(y_test, predictions))
        return svc_model
    
    # Function to apply decision tree
    def dt(self, X_train, X_test, y_train, y_test, max_depth=8):
        dtree = DecisionTreeClassifier( max_depth=max_depth)
        dtree.fit(X_train, y_train)
        predictions = dtree.predict(X_test)
        print(classification_report(y_test, predictions))
        return dtree
    
    # Function to apply random forest
    def rf(self, X_train, X_test, y_train, y_test, n_estimators=100, max_depth=8):
        rfc = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        rfc.fit(X_train, y_train)
        predictions = rfc.predict(X_test)
        print(classification_report(y_test, predictions))
        return rfc
    
    # Function to apply naive bayes
    def nb(self, X_train, X_test, y_train, y_test):
        nb = MultinomialNB()
        nb.fit(X_train, y_train)
        predictions = nb.predict(X_test)
        print(classification_report(y_test, predictions))
        return nb

    def gbc(self, X_train, X_test, y_train, y_test, random_state=1):
        gbc = GradientBoostingClassifier(random_state=random_state)
        gbc.fit(X_train, y_train)
        predicitions = gbc.predict(X_test)
        print(classification_report(y_test, predicitions))
        return gbc

    def abc(self, X_train, X_test, y_train, y_test):
        abc = AdaBoostClassifier()
        abc.fit(X_train, y_train)
        predictions = abc.predict(X_test)
        print(classification_report(y_test, predictions))
        return abc
    
    def ec(self, X_train, X_test, y_train, y_test, voting='hard', random_state=1):
        clf1 = LogisticRegression(random_state=random_state)
        clf2 = RandomForestClassifier(random_state=random_state)
        clf3 = GaussianNB()
        clf4 = SVC(random_state=random_state)
        clf5 = DecisionTreeClassifier(random_state=random_state)
        clf6 = KNeighborsClassifier()
        #clf7 = MLPClassifier(random_state=1)

        estimators = [('lr', clf1), ('rf', clf2), ('gnb', clf3), ('svc', clf4), ('dt', clf5), ('knn', clf6)]

        eclf = VotingClassifier(estimators=estimators, voting=voting)
        eclf.fit(X_train, y_train)
        predictions = eclf.predict(X_test)
        print(classification_report(y_test, predictions))
        return eclf


    # def neural_net(self, ):
    #     model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(96 , activation = 'relu') ,
    #     tf.keras.layers.Reshape((32, 1, 3)),
    #     tf.keras.layers.Conv1D(3, kernel_size=4, strides=2, padding="same" ) ,
    #     tf.keras.layers.BatchNormalization(momentum=0.9),
    #     tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.Dense(10 , activation = 'relu') ,
    #     tf.keras.layers.Dense(10 , activation = 'relu') , 
    #     tf.keras.layers.Reshape((320,)),
    #     tf.keras.layers.Dense(1 , activation = 'sigmoid')
    #     ])


    #     model.compile(loss = tf.keras.losses.binary_crossentropy , 
    #             optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
    #             metrics = ['accuracy'])

    #     return model
    
    # Function to apply cross validation
    def cross_validation(self, model, X, y, cv=5):
        scores = cross_val_score(model, X, y, cv=cv)
        print(scores)
        print(scores.mean())
        return scores

    # Function to apply grid search
    def grid_search(self, model, param_grid, X_train, X_test, y_train, y_test, cv=10):
        grid = GridSearchCV(model, param_grid, cv=cv)
        grid.fit(X_train, y_train)
        grid_predict = grid.predict(X_test)
        print(classification_report(grid_predict, y_test))
        print(grid.best_params_)
        print(grid.best_estimator_)
        return grid
    
    # Function to apply randomised search
    def randomised_search(self, model, X, y, cv=5, n_iter=100, param_grid=None):
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
        predictions = model.predict(X_test)
        #print(classification_report(y_test, predictions))
        print(predictions)
        return model
    
    def plot_confusion_matrix(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        predictions = np.round(predictions)
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True)
        plt.show()
        return cm
    
    def compare_classifier_reports(self, models, X_test, y_test):
        for model in models:
            predictions = model.predict(X_test)
            print(classification_report(y_test, predictions))

    def find_best_model(self, models, X_test, y_test):
        best_model = None
        best_accuracy = 0
        for model in models:
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
        return best_model, best_accuracy

    def model_score(self, model, X_test, y_test):
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)

        return scores

    

    def fit(self, X, y):
        pass

    def predict(self, model, X):
        prediction = model.predict(X)
        return prediction

class Simple_CNN():
    def __init__(self, TRAIN_PATH, VAL_PATH, HEIGHT=224, WIDTH=224, EPOCHS=20, BATCH_SIZE=32, LEARNING_RATE=1e-4, RGB=3) -> None:
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH
        self.EPOCHS = EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        self.LEARNING_RATE = LEARNING_RATE
        self.TRAIN_PATH = TRAIN_PATH
        self.VAL_PATH = VAL_PATH
        self.RGB = RGB


    def preprocess(self):
        train_data = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255, rotation_image=20, zoom_range=0.15,
                                                                     width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                                                                     horizontal_flip=True, fill_mode='nearest')
        train = train_data.flow_from_directory(directory=self.TRAIN_PATH, shuffle=True, target_size=(self.HEIGHT, self.WIDTH), class_mode='binary', batch_size=self.BATCH_SIZE)

        train_ds = image_dataset_from_directory(directory=self.TRAIN_PATH, labels='inferred', label_mode='int', class_names=None,
                                                color_mode='rgb', batch_size=self.BATCH_SIZE, image_size=(self.WIDTH, self.HEIGHT), shuffle=True,
                                                seed=None, validation_split=None, subset=None, interpolation='bilinear', follow_links=False,
                                                crop_to_aspect_ratio=False, pad_to_aspect_ratio=False, data_format=None, verbose=True)

        val_data = tf.keras.preprocessing.image.ImageDataGenerator()
        validation = val_data.flow_from_directory(directory=self.VAL_PATH, shuffle=False, target_size=(self.HEIGHT, self.WIDTH), class_mode='binary', batch_size=self.BATCH_SIZE)

        baseModel = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(self.HEIGHT, self.WIDTH, self.RGB)))
        baseModel.trainable = False

        return train, validation, baseModel
    
    def model(self):
        train, validation, baseModel = self.preprocess()

        model = Sequential([
            Lambda
        ])
    

class CNN():
    def __init__(self, X_test, y_test):
        #self.input_data=input_data
        self.X_test = X_test
        self.y_test = y_test
        self.config = config()
        self.N_VARS = self.config.N_VARS

    def get_model(self):
        model_config = {
                        'input_shape': (32, 32, 3), 
                        'padding': 'same',
                        'pad_out': 2,
                        'N_VARS': 3,
                        'FLUCTUATIONS_PREDICT': True
                        }
        input_shape = model_config['input_shape']
        padding = model_config['padding']
        pad_out = model_config['pad_out']
        N_VARS = model_config['N_VARS']

        input_shape = layers.Input(shape = input_shape, name = 'input_data')
        conv_1 = layers.Conv2D(64, (5, 5), padding=padding, data_format='channels_first')(self.X_test)
        batch_1 = layers.BatchNormalization(axis=1)(conv_1)
        activation_1 = layers.Activation('relu')(batch_1)
        conv_2 = layers.Conv2D(128, (3,3), padding=padding, data_format='channels_first')(activation_1)
        batch_2 = layers.BatchNormalization(axis=1)(conv_2)
        activation_2 = layers.Activation('relu')(batch_2)
        conv_3 = layers.Conv2D(256, (3,3), padding=padding, data_format='channels_first')(activation_2)
        batch_3 = layers.BatchNormalization(axis=1)(conv_3)
        activation_3 = layers.Activation('relu')(batch_3)
        conv_4 = layers.Conv2D(256, (3,3), padding=padding, data_format='channels_first')(activation_3)
        batch_4 = layers.BatchNormalization(axis=1)(conv_4)
        activation_4 = layers.Activation('relu')(batch_4)
        conv_5 = layers.Conv2D(128, (3,3), padding=padding, data_format='channels_first')(activation_4)
        batch_5 = layers.BatchNormalization(axis=1)(conv_5)
        activation_5 = layers.Activation('relu')(batch_5)

        conv_branch1 = layers.Conv2D(1, (3,3), padding=padding, data_format = 'channels_first')(activation_5)
        if (self.config.FLUCTUATIONS_PREDICT == True):
            activation_branch1 = layers.Activation('thres_relu')(conv_branch1)
            output_branch1 = layers.Cropping2D(cropping = ((int(pad_out/2), int(pad_out/2)),
                                                           (int(pad_out/2), int(pad_out/2))),
                                                           data_format='channels_first', name='output_branch1')(activation_branch1)
        else:
            activation_branch1 = layers.Activation('relu')(conv_branch1)
            output_branch1 = layers.Cropping2D(cropping = ((int(pad_out/2), int(pad_out/2)),
                                                           (int(pad_out/2), int(pad_out/2))),
                                                           data_format='channels_first', name='output_branch1')(activation_branch1)
            
        losses = {'output_branch1': 'mse'}

        if (N_VARS == 2):
            conv_branch2 = layers.conv2D(1, (3,3), padding=padding, data_format = 'channels_first')(activation_5)
            if (self.FLUCTUATIONS_PREDICT == True):
                activation_branch2 = layers.Activation('thres_relu')(conv_branch2)
                output_branch2 = layers.Cropping2D(cropping = ((int(pad_out/2), int(pad_out/2)),
                                                            (int(pad_out/2), int(pad_out/2))),
                                                            data_format='channels_first', name='output_branch2')(activation_branch2)
            else:
                activation_branch2 = layers.Activation('relu')(conv_branch2)
                output_branch2 = layers.Cropping2D(cropping = ((int(pad_out/2), int(pad_out/2)),
                                                            (int(pad_out/2), int(pad_out/2))),
                                                            data_format='channels_first', name='output_branch2')(activation_branch2)
                
            losses['output_branch2'] = 'mse'

        elif (N_VARS == 3):
            conv_branch2 = layers.Conv2D(1, (3,3), padding=padding, data_format = 'channels_first')(activation_5)
            if (self.FLUCTUATIONS_PREDICT == True):
                activation_branch2 = layers.Activation('thres_relu')(conv_branch2)
                output_branch2 = layers.Cropping2D(cropping = ((int(pad_out/2), int(pad_out/2)),
                                                            (int(pad_out/2), int(pad_out/2))),
                                                            data_format='channels_first', name='output_branch2')(activation_branch2)
            else:
                activation_branch2 = layers.Activation('relu')(conv_branch2)
                output_branch2 = layers.Cropping2D(cropping = ((int(pad_out/2), int(pad_out/2)),
                                                            (int(pad_out/2), int(pad_out/2))),
                                                            data_format='channels_first', name='output_branch2')(activation_branch2)
                
            losses['output_branch2'] = 'mse'

            conv_branch3 = layers.Conv2D(1, (3,3), padding=padding, data_format = 'channels_first')(activation_5)
            if (self.FLUCTUATIONS_PREDICT == True):
                activation_branch3 = layers.Activation('thres_relu')(conv_branch3)
                output_branch3 = layers.Cropping2D(cropping = ((int(pad_out/2), int(pad_out/2)),
                                                            (int(pad_out/2), int(pad_out/2))),
                                                            data_format='channels_first', name='output_branch3')(activation_branch3)
            else:
                activation_branch3 = layers.Activation('relu')(conv_branch3)
                output_branch3 = layers.Cropping2D(cropping = ((int(pad_out/2), int(pad_out/2)),
                                                            (int(pad_out/2), int(pad_out/2))),
                                                            data_format='channels_first', name='output_branch3')(activation_branch3)

            outputs_model = [output_branch1, output_branch2, output_branch3]

            losses['output_branch3'] = 'mse'

        else:
            outputs_model = output_branch1

        CNN_model = tf.keras.models.Model(inputs=self.config.inputs, threshold=self.config.RELU_THRESHOLD, outputs=outputs_model)
        CNN_model.compile(optimizer='adam', loss=losses)
        return CNN_model, []
    
    def get_dataset(self):
        # Load and preprocess the dataset for training and validation
        # Assuming input_data, X_test, and y_test are part of the dataset

        # Split the dataset into training and validation sets
        dataset_train, dataset_val = train_test_split(self.input_data, test_size=0.2, random_state=42)

        return dataset_train, dataset_val

class svm(ML):
    


    def run(self, X_train, X_test, y_train, y_test, kernel='rbf'):
        svc_model = SVC(kernel=kernel)
        svc_model.fit(X_train, y_train)
        predictions = svc_model.predict(X_test)
        print(classification_report(y_test, predictions))
        return svc_model

class ffnn(BasePredictor):
    def __init__(self, hidden_layers = [], dropout = 0, epochs = 5, activation = [],batch_size = None):
        """
        Initialises the ffnn class for creating a feedforward neural-network.

        Args:
            hidden_layers (list, optional): List of hidden layer sizes. Defaults to [].
            dropout (float, optional): Dropout rate for regularisation. Defaults to 0.
            epochs (int, optional): Number of training epochs. Defaults to 5.
            activation (list, optional): List of activation functions for hidden layers. Defaults to [].
            batch_size (int, optional): Batch size for training. Defaults to None.
        """
        # Store Hyperparameters
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.epochs = epochs
        self.activation = activation
        self.batch_size = batch_size

        
        # Create the Keras Sequential model
        self.model = Sequential()
        # Add hidden layers
        for i in range(len(self.hidden_layers)):
            if i == 0:
                self.model.add(Dense(self.hidden_layers[i], activation = self.activation[i], input_dim = self.hidden_layers[i]))
            else:
                self.model.add(Dense(self.hidden_layers[i], activation = self.activation[i]))
            if self.dropout > 0:
                self.model.add(Dropout(self.dropout))

        # add the output layer
        self.model.add(Dense(1, activation = 'sigmoid'))
        # set the loss function and optimiser and then finally compile the model
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
# print(data)
data = pd.DataFrame(data)
data['target'] = np.random.randint(0, 2, 1000)
# print(data.head())