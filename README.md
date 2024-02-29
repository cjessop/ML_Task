![example workflow](https://img.shields.io/badge/HEIMDALL-yellow) ![example workflow](https://img.shields.io/badge/build-passing-green) ![example_workflow](https://img.shields.io/badge/version-0.2-blue)
 ![example workflow](https://img.shields.io/badge/copyright-all%20rights%20reserved-darkred) 
 
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

# HEIMDALL v0.2 - Updated 29/02/24
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
Consider using GridSearchCV or RandomizedSearchCV for more structured hyperparameter tuning.
Explore model explainability techniques (e.g., SHAP, LIME) to understand model behavior better.
Employ techniques for handling imbalanced datasets if applicable.
Consider model deployment strategies for real-world applications.


Copyright © 2024 <C Jessop>. All rights reserved.
