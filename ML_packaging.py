
from audioop import cross
from BaseMLClasses import BasePredictor
from BaseMLClasses import ML
from BaseMLClasses import ffnn
import pickle
import os
import numpy as np


# Create meta class to apply all machine learning algorithms
class ML_meta:
    """
    A meta class that handles the application of all ML models. 
    The current models are:
    - Support Vector Machine
    - Naive Bayes
    - Decision Tree
    - Logistic Regression
    - Multi-Layered Perceptron
    - Random Forest
    - k-Nearest-Neighbour
    - Ensemble Classifier (all models combined)

    Includes the call to instantiate the ML class and apply test-train split

    arguments: 
    data - input dataset in disordered format - Column labelled dataset
    ffnn - whether usage of the feed-forward neural network to make a prediction - True or False
    all - whether or not to apply all classifier models to the singe dataset - True or False
    model - the name of the model to be applied - String
    model_dict - dictionary of all models and their corresponding names
    target - the name of the target feature from input dataset - String
    help - whether or not to print the help message - True or False
    clean - whether or not to delete all saved models - True or False
    search - perform grid search either randomly or evenly spaced on a grid - String
    
    output:
    None


    """
    def __init__(self, data, ffnn=False, all=True, model=False, model_dict={
                                        "SupportVector": "SVM",
                                        "KNearestNeighbour": "kNN",
                                        "LinearRegression": "LinReg",
                                        "NaiveBayes": "NB",
                                        "MultiLayerPerceptron": "MLP",
                                        "DecisionTree": "DT",
                                        "RandomForest": "RF",
                                        "NeuralNetwork": "NN",
                                        "EnsembleClassifier": "EC"
                                    }, target='target', help=False, clean=False, search=None, cross_val=False):
        self.data = data
        self.ffnn = ffnn
        self.all = all
        self.model = model
        self.model_dict = model_dict
        self.target = target
        self.help = help
        self.clean = clean
        self.search = search
        self.cross_val = cross_val

    def misc(self):
        if self.help is True:
            print("This is a meta class that handles the application of all ML models. The current models are: Support Vector Machine, \
                  Naive Bayes, Decision Tree, Logistic Regression, Multi-Layered Perceptron, Random Forest, k-Nearest-Neighbour, Ensemble Classifier (all models combined). \
                  Includes the call to instantiate the ML class and apply test-train split")

        if self.clean is True:
            delete_var = input("Are you sure you want to delete all saved models? (y/n)")
            if delete_var == "y" or delete_var == "Y":
                print("Deleting saved models")
                # Delete any saved models inclduing all files that end in .pkl
                for filename in os.listdir():
                    if filename.endswith(".pkl"):
                        os.remove(filename)
                    else:
                        continue
            else:
                print("Not deleting saved models")
                pass

    # Call the ML class to apply all machine learning algorithms
    def call_ML(self):
        ml = ML(self.data)
        return ml


    def split_data(self, encode_categorical=True, y='target'):
        ml = self.call_ML()
        X, y = ml.split_X_y(self.target)
        if encode_categorical is True:
            X, y = ml.encode_categorical(X, y)

        return X, y

    def apply_all_models(self, flag=False):
        ml = self.call_ML()
        X, y = self.split_data(encode_categorical=False)
        X_train, X_test, y_train, y_test = self.call_ML().split_data(X, y)
        if flag == False:
            pass
        else:
            ml = self.call_ML()
            #Apply test train split
            X, y = self.split_data(self.data)

            rf = ml.random_forest(X_train, X_test, y_train, y_test)
            svm = ml.svm(X_train, X_test, y_train, y_test)
            knn = ml.knn(X_train, X_test, y_train, y_test)
            lr = ml.logistic_regression(X_train, X_test, y_train, y_test)
            nb = ml.naive_bayes(X_train, X_test, y_train, y_test)
            mlp = ml.mlp(X_train, X_test, y_train, y_test)
            dt = ml.decision_tree(X_train, X_test, y_train, y_test)
            nn = ml.nn(X_train, X_test, y_train, y_test)
            ec = ml.ensemble_classifier(X_train, X_test, y_train, y_test, voting='hard')
            
            #ml.ffnn(X_train, X_test, y_train, y_test)
            #ml.nn(X_train, X_test, y_train, y_test)

            #Compare the results of all models

            



    def apply_neural_net(self):
        if self.ffnn:
            ffnn_predictor = ffnn(3, activation='sigmoid', batch_size=5)
            ml_obj = ML(self.data)
            x, Y = ml_obj.split_X_y(X='target', y='target')
            X_train, X_test, y_train, y_test = ml_obj.split_data(x, Y)

            ffnn_predictor.fit(X_train, y_train)
            ffnn_predictor.predict(X_test)

    def apply_single_model(self, cm=False, save_model=False, save_model_name=False):
        X, y = self.split_data(encode_categorical=False)
        X_train, X_test, y_train, y_test = self.call_ML().split_data(X, y)
        self.model_dict = {
                                        "SupportVector": "SVM",
                                        "KNearestNeighbour": "kNN",
                                        "LinearRegression": "LinReg",
                                        "NaiveBayes": "NB",
                                        "MultiLayerPerceptron": "MLP",
                                        "DecisionTree": "DT",
                                        "RandomForest": "RF",
                                        "NeuralNetwork": "NN",
                                        "EnsembleClassifier": "EC"
                                    }

        model_list = []
        model_list.append(self.model)
        if self.model is not False:
            ml_single_model = ML(self.data)
            self.model_dict = {
                                        "SVM": ml_single_model.svm, #"SVM",
                                        "KNN": ml_single_model.knn,
                                        "LR": ml_single_model.lr,
                                        "NB": ml_single_model.nb,
                                        "MLP": ml_single_model.mlp,
                                        "DT": ml_single_model.dt,
                                        "RF": ml_single_model.rf,
                                        #"NN": ml_single_model.nn,
                                        "EC": ml_single_model.ec
                                    }
            if self.model in self.model_dict.keys():
                print("Selected single model is " + str(self.model_dict[self.model]))
                model = self.model_dict[self.model](X_train, X_test, y_train, y_test)
                if self.search is not None:
                    if self.model == "SVM":
                        param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
                        
                    elif self.model == "KNN":
                        param_grid = { 'n_neighbors' : [5,7,9,11,13,15],
                                        'weights' : ['uniform','distance'],
                                        'metric' : ['minkowski','euclidean','manhattan']}

                    elif self.model == "NB":
                        param_grid = { 'var_smoothin' : np.logspace(0, 9, num=100)}

                    elif self.model == "RF":
                        param_grid = { 'n_estimators': [25, 50, 100, 150, 200],
                                        'max_features': ['auto', 'sqrt', 'log2', None],
                                        'max_depth': [3, 5, 7, 9, 11] }

                    elif self.model == "DT":
                        param_grid = { 'max_features': ['auto', 'sqrt'],
                                        'max_depth': 8 }

                    elif self.model == "LR":
                        param_grid = { 'solver' : ['lbfgs', 'sag', 'saga', 'newton-cg'] }

                if self.search == "random":
                    ml_single_model.randomised_search(model, X_train, y_train, param_grid=param_grid)
                elif self.search == "grid":
                    ml_single_model.grid_search(model, param_grid, X_train, X_test, y_train, y_test, cv=10)
                    

                elif self.cross_val is not False:
                    ml_single_model.cross_validation(model, X_train, y_train)  
                # else:
                #     model = self.model_dict[self.model](X_train, X_test, y_train, y_test)
                if save_model is True:
                    pickle.dump(model, open(save_model_name, 'wb'))
                if cm is True:
                    ML.plot_confusion_matrix(self, model, X_test, y_test)

                
                        
        self.misc()


    # Call the deep learning class to apply all deep learning algorithms
    # def call_DL(self):
    #     dl = DL(self.data)


if __name__ == "__main__":
        # Initialise the meta class
    meta_obj = ML_meta(data, all=False, model="MLP")
    meta_obj.apply_single_model()
    

