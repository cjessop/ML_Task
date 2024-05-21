![example workflow](https://img.shields.io/badge/HEIMDALL-yellow) ![example workflow](https://img.shields.io/badge/build-passing-green) ![example_workflow](https://img.shields.io/badge/version-0.3-blue)
 ![example workflow](https://img.shields.io/badge/copyright-all%20rights%20reserved-darkred) 
 
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

# HEIMDALL v0.3 - Updated 17/04/24
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

-`__init__(self, data)`: Initialse the ML class with the given `data`
-`split_X_y(self, y`: Splits the data into features `X`, and target variable `y`
-`encode_categorical(self, X, y)`: Encodes the categorical data in `X` and `y`
-`missing_data(self, X, y, strategy='mean')`: Handles any missing data in `X` and `y` using the specified strategy
-`extract_features(self, X, y, test_size=0.2)`: Extracts features from classification data and splits into training and test sets
-`split_data(self, X, y, test_size=0.2)`: Splits data into training and testing sets
-`scale_data(self, X_train, X_test)`: Scales the training and test data
-`prepare_data(self, name, label, columns, end_range, index_column=False)`: Prepares the data for analysis
-`lr(self, X_train, X_test, y_train, y_test)`: Applies the logistic regression algorithm to the data
-`knn(self, X_train, X_test, y_train, y_test, n_neighbors=1)`: Applies the K-Nearest Neighbors algorithm to the data
-`svm(self, X_train, X_test, y_train, y_test, kernel='rbf')`: Applies the Support Vector Machine algorithm to the data
-`dt(self, X_train, X_test, y_train, y_test, max_depth=8)`: Applies the Decision Tree algorithm to the data
-`rf(self, X_train, X_test, y_train, y_test, n_estimators=100, max_depth=8)`: Applies the Random Forest algorithm to the data
-`nb(self, X_train, X_test, y_train, y_test)`: Applies the Naive Bayes algorithm to the data
-`gbc(self, X_train, X_test, y_train, y_test, random_state=1)`: Applies the Gradient Boosting Classifier algorithm to the data
-`abc(self, X_train, X_test, y_train, y_test)`: Applies the AdaBoost Classifier algorithm to the data
-`ec(self, X_train, X_test, y_train, y_test, voting='hard', random_state=1)`: Applies the Ensemble Classifier (combination of multiple models) to the data
-`cross_validation(self, model, X, y, cv=5)`: Applies K-fold cross-validation to the given model
-`grid_search(self, model, param_grid, X_train, X_test, y_train, y_test, cv=10)`: Performs a grid search for hyperparameter tuning
-`randomised_search(self, model, X, y, cv=5, n_iter=100, param_grid=None)`: Performs a randomised search for hyperparameter tuning
-`mlp(self, X_train, X_test, y_train, y_test, hidden_layers=1, neurons=8, activation='relu', optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], epochs=25, batch_size=32, validation_split=0.2, verbose=1)`: Applies a multi-layer perceptron (MLP) neural -network to the data
-`plot_confusion_matrix(self, model, X_test, y_test)`: Plots the confusion matrix for the selected model and data
-`compare_classifier_reports(self, models, X_test, y_test)`: Compares the classification reports for multiple models
-`find_best_model(self, models, X_test, y_test)`: Finds the best model among the given models based on accuracy
-`model_score(self, model, X_test, y_test)`: Calculates the cross-validation scores for the given model and data




Copyright © 2024 <C Jessop>. All rights reserved.
