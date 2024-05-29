![example workflow](https://img.shields.io/badge/HEIMDALL-yellow) ![example workflow](https://img.shields.io/badge/build-passing-green) ![example_workflow](https://img.shields.io/badge/version-0.3-blue)
 ![example workflow](https://img.shields.io/badge/copyright-all%20rights%20reserved-darkred) 
 
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

# HEIMDALL v0.3 - Updated 21/05/2024
## High-Resolution [----] Identification and [----] Discriminator [----]

### 1. ML_meta Class:

Purpose: Acts as a coordinator, managing the application of various machine learning algorithms.
Key Methods:
apply_all_models(flag=True): Applies multiple ML models and compares their scores.
apply_neural_net(): Applies a feedforward neural network (FFNN).
apply_single_model(cm=False, save_model=False, save_model_name=False): Applies a specified model, with options for confusion matrix plotting and model saving.
split_data(encode_categorical=True, y='target'): Splits data into features (X) and target (y), with optional encoding of categorical features.
call_ML(): Instantiates the ML class to access model implementation details.

### 2. ML Class:

Houses the implementation of various machine learning algorithms, including:
- `Support Vector Machine (SVM)`
- `Naive Bayes (NB)`
- `Decision Tree (DT)`
- `Random Forest (RF)`
- `k-Nearest-Neighbour (kNN)`
- `Logistic Regression (LR)`
- `Multi-Layered Perceptron (MLP)`
- `Ensemble Classifier (EC)`
- `Gradient Boosted Classifier (GBC)`
- `Ada Boosted Classifier (ABC)`

PLANNED: 
MultiLayeredPerceptron (MLP) bug fixes

### 3. FFNN Class:

Implements a feedforward neural network.

### 4. BasePredictor Class:

Provides a base class for prediction-related functionality.

### 5. CNN Class:

Implements a convoluted neural network architecture.

### 6. YOLOv8 Object identifier

Provides the functionality for a user-trained YOLO identifier to predict on either pre-recorded or live video stream

### 7. Pipeline Class:

Reads in YAML configuration file to streamline the implementation and usage of the package

--------------------------------------------------------------------------------------------------
## Usage

### Use case 1: Applying Multiple Models and Comparing Performance:
import ML_meta

model_coordinator = ML_meta.ML_meta()  # Instantiate the coordinator
model_coordinator.apply_all_models(flag=True)  # Apply all models and generate a report

### Use case 2: Applying a Specific Model with Confusion Matrix and Model Saving:

model_coordinator.apply_single_model("RF", cm=True, save_model=True, save_model_name="best_rf.pkl")

### Use case 3: Applying a Feedforward Neural Network:

model_coordinator.apply_neural_net()

### Use case 4: Customising Model Application:

Split data with custom settings
X_train, X_test, y_train, y_test = model_coordinator.split_data(encode_categorical=False, y="my_target")

Access specific models and control parameters
ml_instance = model_coordinator.call_ML()
ml_instance.apply_decision_tree(X_train, X_test, y_train, y_test, max_depth=5)

The ML_meta class serves as the primary entry point for users.
Methods like apply_all_models, apply_single_model, and apply_neural_net provide streamlined model application.
Access to individual models and their parameters is available through call_ML.
Customise data splitting and encoding using split_data.

--------------------------------------------------------------------------------------------------

## TO-DO
- `Refine YAML input capability`
- `Fully refine the Pipeline class to be fully functional`
- `Explore model explainability techniques (e.g., SHAP, LIME) to understand model behaviour better`
- `Employ techniques for handling imbalanced datasets if applicable`
- `Consider model deployment strategies for real-world applications`
- `Add a Setup.cfg file`
- `Add a Setup.py file`
- `Add UnitTests for the package`

The BaseMLClasses module contains several classes and functions related to machine learning tasks, such as data
processing, model fitting, and prediction. This module provides the framework for applying various machine learning
algorithms, including classical methods like logistic regression, decision trees, and support vector machines. Additionally, 
it provides deep learning methods such as convolution neural networks.

## Classes

**Base predictor**

`class BasePredictor(ABC)`

This is an abstract base class that defines the core functionality expected from predictive models, such as fitting and 
predicting. This class provides utility methods for parameter management and model resetting/loading.

### Methods

- `fit(self, X, y)`: Trains model on the given input data `X` and the target `y`. This method is abstract and must implemented
  by subclasses
- `predict(self, X`: Makes a prediction on the new data `X`, using the trained model. This is an abstract method and must be
  implemented by the subclass
- `_get_params_names(cls)`: This retrieves the names of the parameters for the class's constructor. This is a class method
- `get_param_names(self, deep=True)`: Returns a dict of the class's hyperparameters
- `reset(self)`: Creates a new instance of the class with the same pararmeters as the current instance
- `load_params(self, params=None)`: Loads new parameters into the classes instance

**ML**

`class ML(BasePredictor)`

This class initialises the ML class with the given data and provides the methods for data preprocessing, model fitting, and
prediction.

### Methods 

- `__init__(self, data)`: Initialse the ML class with the given `data`
- `split_X_y(self, y`: Splits the data into features `X`, and target variable `y`
- `encode_categorical(self, X, y)`: Encodes the categorical data in `X` and `y`
- `missing_data(self, X, y, strategy='mean')`: Handles any missing data in `X` and `y` using the specified strategy
- `extract_features(self, X, y, test_size=0.2)`: Extracts features from classification data and splits into training and test sets
- `split_data(self, X, y, test_size=0.2)`: Splits data into training and testing sets
- `scale_data(self, X_train, X_test)`: Scales the training and test data
- `prepare_data(self, name, label, columns, end_range, index_column=False)`: Prepares the data for analysis
- `lr(self, X_train, X_test, y_train, y_test)`: Applies the logistic regression algorithm to the data
- `knn(self, X_train, X_test, y_train, y_test, n_neighbors=1)`: Applies the K-Nearest Neighbors algorithm to the data
- `svm(self, X_train, X_test, y_train, y_test, kernel='rbf')`: Applies the Support Vector Machine algorithm to the data
- `dt(self, X_train, X_test, y_train, y_test, max_depth=8)`: Applies the Decision Tree algorithm to the data
- `rf(self, X_train, X_test, y_train, y_test, n_estimators=100, max_depth=8)`: Applies the Random Forest algorithm to the data
- `nb(self, X_train, X_test, y_train, y_test)`: Applies the Naive Bayes algorithm to the data
- `gbc(self, X_train, X_test, y_train, y_test, random_state=1)`: Applies the Gradient Boosting Classifier algorithm to the data
- `abc(self, X_train, X_test, y_train, y_test)`: Applies the AdaBoost Classifier algorithm to the data
- `ec(self, X_train, X_test, y_train, y_test, voting='hard', random_state=1)`: Applies the Ensemble Classifier (combination of multiple models) to the data
- `cross_validation(self, model, X, y, cv=5)`: Applies K-fold cross-validation to the given model
- `grid_search(self, model, param_grid, X_train, X_test, y_train, y_test, cv=10)`: Performs a grid search for hyperparameter tuning
- `randomised_search(self, model, X, y, cv=5, n_iter=100, param_grid=None)`: Performs a randomised search for hyperparameter tuning
- `mlp(self, X_train, X_test, y_train, y_test, hidden_layers=1, neurons=8, activation='relu', optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], epochs=25, batch_size=32, validation_split=0.2, verbose=1)`: Applies a multi-layer perceptron (MLP) neural -network to the data
- `plot_confusion_matrix(self, model, X_test, y_test)`: Plots the confusion matrix for the selected model and data
- `compare_classifier_reports(self, models, X_test, y_test)`: Compares the classification reports for multiple models
- `find_best_model(self, models, X_test, y_test)`: Finds the best model among the given models based on accuracy
- `model_score(self, model, X_test, y_test)`: Calculates the cross-validation scores for the given model and data

**Simple CNN**

`class Simple_CNN`

This class implements a simple convolution neural network (CNN) architecture for the purposes of image classification.

### Methods

- `__init__(self, TRAIN_PATH, VAL_PATH, HEIGHT=224, WIDTH=224, EPOCHS=20, BATCH_SIZE=32, LEARNING_RATE=1e-4, RGB=3)`: Initialises the Simple_CNN class with the given parameters
- `preprocess(self)`: Preprocesses the data for training and validation
- `model(self, save=False)`: Builds and trains the CNN model

**CNN**

`class CNN()`

This class implements a simple convolution neural network (CNN) architecture for the purposes of image classification. Differs from the `Simple_CNN` class as it has a wider 
range of architectures (shape wise).

### Methods

- `__init__(self, X_test, y_test)`: Initialises the CNN class with the given test data
- `get_model(self)`: Builds and returns the CNN model
- `get_dataset(self)`: Loads and preprocesses the dataset for training and validation

**SVM** 

`class svm(ML)`

This class is a subclass of `ML`, it provides methods for applying the Support Vector Machine (SVM) algorithm.

### Methods

- `run(self, X_train, X_test, y_train, y_test)`: This trains and evaluates the SVM model on the input data

**ffnn** 

`class ffnn(BasePredictor)`

This class implements a feed-forward neural network (ffnn) for classification (or regression?) tasks. Inherits
from BasePredictor.

### Methods

- `__init__(self, hidden_layers=[], dropout=0, epochs=5, activation=[], batch_size=None)`: Initialises the ffnn class with the given hyperparameters
- `fit(self, X, y)`: Trains the FFNN model on the given data
- `predict(self, X)`: Makes predictions using the trained FFNN model

**Pipeline Loader**

`class PipelineLoader(yaml.safeloader)`

This class is a custom YAML loader which provides functionality for loading and constructing objects from YAML files.

### Methods

- `load(cls, instream)`: Loads and returns the single data object from the input stream
- `construct_map(self, node, deep=False)`: Constructs a mapping object from the given YAML node
- `construct_ref(self, node)`: Constructs a reference object from the given YAML node
- `construct_call(self, name, node)`: Constructs a callable object from the given YAML node
- `load_pipeline_yaml(filename)`: Loads a pipeline YAML file and returns its contents

The ML_packaging module contains several classes and functions related to applying the classes and methods found in BaseMLClasses.
This module provides the framework for applying various machine learning algorithms to actual problems, in addition to functionality
such as creating and saving the serialised models themselves. 

**ML_meta**

`class ML_meta()`

This class creates the meta class that holds the functionality to apply all of the machine learning algorithms. It acts as a coordiantor,
interacting with other classes for specific model implementations.

**Parameters**

- data: input dataset in disordered format - Column labelled dataset
- ffnn: bool, optional
  Whether to use a feed-forward neural network to make a prediction. Default is False.
- all: bool, optional
Whether to apply all classifier models to the dataset. Default is True.
- model: str, optional
The name of the model to be applied. Default is False.
- model_dict: dict, optional
Dictionary of all models and their corresponding names. Default is a pre-defined dictionary of model names.
- target: str, optional
The name of the target feature from the input dataset. Default is 'target'.
- help: bool, optional
Whether to print the help message. Default is False.
- clean: bool, optional
Whether to delete all saved models. Default is False.
- search: str, optional
Perform grid search either randomly or evenly spaced on a grid. Options are 'random' or 'grid'. Default is None.
- cross_val: bool, optional
Perform k-fold cross validation. Default is False.
- CNN: bool, optional
Apply a convolutional neural network. Default is None.
- on_GPU: bool, optional
Run the CNN on a GPU. Default is False.
- YOLO: bool, optional
Instantiate an instance of the YOLO class for training or prediction. Default is False.
- data_path: str, optional
The path to the dataset. Default is None.
- image_path: str, optional
The path to the image library. Default is None.
- image_number: int, optional
Number of images to use. Default is None.
- YOLO_model: str, optional
The path or name of the trained YOLO algorithm. Default is None.
- video_path: str, optional
The path to the video on which you would like to predict. Default is None.
- video_capture: bool, optional
Use a connected image or camera sensor for live input to predict on. Default is False.
- YOLO_train: bool, optional
Flag to train a new YOLO model, requires data_path and image_path to be not None. Default is False.
- YOLO_save: bool, optional
Flag to save the YOLO model. Default is False.

**Methods**

- `misc()`: Handles miscellaneous tasks such as printing help messages and cleaning saved models

- `call_ML()`: Creates an instance of the ML class

- `split_data(encode_categorical=True, y='target')`: Splits data into features (X) and target (y), with optional encoding of categorical features

- `apply_all_models(flag=False)`: Applies multiple machine learning models to the dataset and compares their scores

- `apply_neural_net()`: Applies a feedforward neural network model (FFNN)

- `apply_CNN()`: Applies the Convolutional Neural Network architecture defined in BaseMLClasses

- `apply_single_model(cm=False, save_model=False, save_model_name=False)`: Applies a single machine learning model to the dataset

- `apply_YOLO()`: Applies or trains a YOLO model on an input video or live capture

**ML_post_process**

This is a class that handles the post-processing functionality of any saved ML models

**Parameters**

- data: Input dataframe in the same format as the data used to test/train the model.
- saved_model: str, optional
Input model saved as .pkl - Binary Machine Learning Model string name. Default is None.
- predict: bool, optional
Whether or not to predict on input data. Default is False.
- target: str, optional
The name of the target feature. Default is None.
- con_cols: str or list of str, optional
The continuous column names. Default is None.
- feature: str, optional
Specific feature for univariate analysis. Default is None.

**Methods**

- `split_data(encode_categorical=True, y='target')`: Splits data into features (X) and target (y), with optional encoding of categorical features

- `get_X_test()`: Gets the X_test portion of the dataset from the split data method

- `load_and_predict()`: Loads a saved serialized trained ML/AI model and makes predictions on a set of input variables

- `data_info()`: Outputs various information on the dataset, such as shape, values, and unique counts

- `target_plot()`: Plots the target variable output as a bar graph

- `corr_plot()`: Plots the correlation between parameters of the input dataset

Copyright © 2024 <C Jessop>. All rights reserved.
