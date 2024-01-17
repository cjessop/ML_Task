
from BaseMLClasses import BasePredictor
from BaseMLClasses import ML
from BaseMLClasses import ffnn


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
    
    output:
    None


    """
    def __init__(self, data, ffnn=False, all=True, model=False, model_dict={}, target='target'):
        self.data = data
        self.ffnn = ffnn
        self.all = all
        self.model = model
        self.model_dict = model_dict
        self.target = target


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
            

            ml.random_forest(X_train, X_test, y_train, y_test)
            ml.svm(X_train, X_test, y_train, y_test)
            ml.logistic_regression(X_train, X_test, y_train, y_test)
            ml.knn(X_train, X_test, y_train, y_test)
            ml.naive_bayes(X_train, X_test, y_train, y_test)
            ml.decision_tree(X_train, X_test, y_train, y_test)
            ml.mlp(X_train, X_test, y_train, y_test)
            ml.ensemble_classifier(X_train, X_test, y_train, y_test, voting='hard')
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

    def apply_single_model(self):
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
            if self.model in self.model_dict.values():
                print("Selected single model is " + str(self.model))

                #try:
                if self.model == "SVM":
                    ml_single_model.svm(X_train, X_test, y_train, y_test)
                elif self.model == "kNN":
                    ml_single_model.knn(X_train, X_test, y_train, y_test)
                elif self.model == "LinReg":
                    ml_single_model.logistic_regression(X_train, X_test, y_train, y_test)
                elif self.model == "NB":
                    ml_single_model.naive_bayes(X_train, X_test, y_train, y_test)
                elif self.model == "MLP":
                    ml_single_model.mlp(X_train, X_test, y_train, y_test)
                elif self.model == "DT":
                    ml_single_model.decision_tree(X_train, X_test, y_train, y_test)
                elif self.model == "RF":
                    ml_single_model.random_forest(X_train, X_test, y_train, y_test)
                elif self.model == "NN":
                    ml_single_model.mlp(X_train, X_test, y_train, y_test)
                elif self.model == "EC":
                    ml_single_model.ensemble_classifier(X_train, X_test, y_train, y_test, voting='hard')
                #except:
                #    print("Error in applying single model")
                #    pass


    # Call the deep learning class to apply all deep learning algorithms
    # def call_DL(self):
    #     dl = DL(self.data)


if __name__ == "__main__":
        # Initialise the meta class
    meta_obj = ML_meta(data, all=False, model="MLP")
    meta_obj.apply_single_model()
    

