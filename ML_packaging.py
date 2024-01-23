# HORUS - High-Precision Object Classification and Radar Unidentified Signal Separation

from audioop import cross
from BaseMLClasses import BasePredictor
from BaseMLClasses import ML
from BaseMLClasses import ffnn
import pickle
import os
import numpy as np
import pandas as pd
#import plotly.express as px
import matplotlib.pyplot as plt
import warnings
import glob
import seaborn as sns

warnings.filterwarnings("ignore")

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
    - Gradient Boosted Classifier
    - Ada Boosted Classifier

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
    cross_val - perform k-fold cross validation - True or False
    
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
                  Includes the call to instantiate the ML class and apply test-train split \n")
            print(ML_meta.__doc__)

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

            rf = ml.rf(X_train, X_test, y_train, y_test)
            svm = ml.svm(X_train, X_test, y_train, y_test)
            knn = ml.knn(X_train, X_test, y_train, y_test)
            lr = ml.lr(X_train, X_test, y_train, y_test)
            nb = ml.nb(X_train, X_test, y_train, y_test)
            #mlp = ml.mlp(X_train, X_test, y_train, y_test)
            dt = ml.dt(X_train, X_test, y_train, y_test)
            #nn = ml.nn(X_train, X_test, y_train, y_test)
            ec = ml.ec(X_train, X_test, y_train, y_test, voting='hard')
            gbc = ml.gbc(X_train, X_test, y_train, y_test)
            abc = ml.abc(X_train, X_test, y_train, y_test)
            
            models = [rf, svm, knn, lr, nb, dt, ec, gbc, abc]

            #ml.ffnn(X_train, X_test, y_train, y_test)
            #ml.nn(X_train, X_test, y_train, y_test)

            #Compare the results of all models
            scores = []
            for model in models:
                score = ml.model_score(model, X_test, y_test)
                scores.append(score)

            print(scores)



        return rf, svm, knn, lr, nb, dt, ec, gbc, abc, scores

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
                                        "EnsembleClassifier": "EC",
                                        "GradientBoosted" : "GBC",
                                        "AdaBooster": "ABC"
                                    }

        model_list = []
        model_list.append(self.model)
        if self.model is not False:
            ml_single_model = ML(self.data)
            self.model_dict = {
                                        "SVM": ml_single_model.svm,
                                        "KNN": ml_single_model.knn,
                                        "LR": ml_single_model.lr,
                                        "NB": ml_single_model.nb,
                                        "MLP": ml_single_model.mlp,
                                        "DT": ml_single_model.dt,
                                        "RF": ml_single_model.rf,
                                        #"NN": ml_single_model.nn,
                                        "EC": ml_single_model.ec,
                                        "GBC": ml_single_model.gbc,
                                        "ABC": ml_single_model.abc
                                    }
            if self.model in self.model_dict.keys():
                print("Selected single model is " + str(self.model_dict[self.model]))
                model = self.model_dict[self.model](X_train, X_test, y_train, y_test)
                if self.search is not None:
                    if self.model == "SVM":
                        param_grid = {  'C': [0.1, 1, 10, 100, 1000], 
                                        'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
                                        'kernel': ['rbf']}
                        
                    elif self.model == "KNN":
                        param_grid = { 'n_neighbors' : [5, 7, 9, 11, 13, 15],
                                        'weights' : ['uniform', 'distance'],
                                        'metric' : ['minkowski', 'euclidean', 'manhattan']}

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

                    elif self.model == "GBC":
                        param_grid = { 'n_estimators': [25, 50, 100, 150, 200],
                                        'max_features': ['auto', 'sqrt', 'log2', None],
                                        'max_depth': [3, 5, 7, 9, 11] }

                    elif self.model == "ABC":
                        param_grid = { 'n_estimators': [25, 50, 100, 150, 200, 500],
                                        'algorithm': ['SAMME', 'SAMME.R', None],
                                        'learning_rate': [3, 5, 7, 9, 11], }
                                        #'max_depth': [1, 3, 5, 7, 9, 11] }

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

class ML_post_process(ML_meta):
    """
    A class that handles the post-processing functionality of any saved ML models.

    arguments: 
    model - Input model saved as .pkl - Binary Machine Learning Model string name
    data - Input dataframe in the same format as the data used to test to train the model, i.e. the same labelled columns
    predict - Whether or not to predict on input data - Boolean True or False
    target - The name of the target feature - String
    con_cols - The continuous column names - String or list of strings

    univariate analysis - method that takes a string to perform exploratory data analysis on an input data set. string inputs include:
        - 'output' plots the target variable output as a bar graph
        - 'corr' plots the correlation matrices between features
        - 'pair' plots the pairwise relationships in the input dataset
        - 'kde' kernel density estimate plot of a feature against the target - input string is the name of the feature
    
    
    output:
    None

 
    """
    def __init__(self, data, saved_model=None, predict=False, target=None, con_cols=None, feature=None):
        self.saved_model = saved_model
        #self.X_test = X_test
        self.predict = predict
        self.data= data
        self.target = target
        self.con_cols = con_cols
        self.feature = feature

    def split_data(self, encode_categorical=True, y='target'):
        ml = self.call_ML()
        X, y = ml.split_X_y(self.target)
        if encode_categorical is True:
            X, y = ml.encode_categorical(X, y)

        return X, y

    def get_X_test(self):
        X, y = self.split_data()
        #X, y = self.split_data(encode_categorical=False)
        _, X_test, _, _ = self.call_ML().split_data(X, y)

        return X_test

    def load_and_predict(self): 

        if self.saved_model is not None:
            
            cwd = os.getcwd()
            path = str(cwd)
            pickled_model = pickle.load(open(self.model, 'rb'))

        for filename in os.listdir():
                try:
                    if filename.endswith(".pkl"):
                        file = str(glob.glob('*.pkl')[0])
                        pickled_model = pickle.load(open(file, 'rb'))
                    else:
                        continue

                except:
                    print("Error loading " + str(self.model) + " machine learning model")

        if self.predict == True:
            X_test = self.get_X_test()
            print(X_test)
            print(pickled_model.predict(X_test))

    def data_info(self):
        print("The shape of the dataset is " + str(self.data.shape))
        print(self.data.head())
        dict = {}
        for i in list(self.data.columns):
            dict[i] = self.data[i].value_counts().shape[0]

        print(pd.DataFrame(dict, index=['Unique count']).transpose())
        print(self.data.describe().transpose())

    def target_plot(self):
            fig = plt.figure(figsize=(18,7))
            gs =fig.add_gridspec(1,2)
            gs.update(wspace=0.3, hspace=0.3)
            ax0 = fig.add_subplot(gs[0,0])
            ax1 = fig.add_subplot(gs[0,1])

            background_color = "#ffe6f3"
            color_palette = ["#800000","#8000ff","#6aac90","#da8829"]
            fig.patch.set_facecolor(background_color) 
            ax0.set_facecolor(background_color) 
            ax1.set_facecolor(background_color) 

            # Title of the plot
            ax0.text(0.5,0.5,"Target Count\n",
                    horizontalalignment = 'center',
                    verticalalignment = 'center',
                    fontsize = 20,
                    fontweight='bold',
                    fontfamily='serif',
                    color='#000000')

            ax0.set_xticklabels([])
            ax0.set_yticklabels([])
            ax0.tick_params(left=False, bottom=False)

            # Target Count
            ax1.text(0.35,177,"Output",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
            ax1.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
            sns.countplot(ax=ax1, data = self.data, x = self.target, palette=["#8000ff","#da8829"])
            ax1.set_xlabel("")
            ax1.set_ylabel("")
            #ax1.set_xticklabels([" "])

            ax0.spines["top"].set_visible(False)
            ax0.spines["left"].set_visible(False)
            ax0.spines["bottom"].set_visible(False)
            ax0.spines["right"].set_visible(False)
            ax1.spines["top"].set_visible(False)
            ax1.spines["left"].set_visible(False)
            ax1.spines["right"].set_visible(False)

            plt.show()

    def corr_plot(self):
        df_corr = self.data[self.con_cols].corr().transpose()
        df_corr
        fig = plt.figure(figsize=(10,10))
        gs = fig.add_gridspec(1,1)
        gs.update(wspace=0.3, hspace=0.15)
        ax0 = fig.add_subplot(gs[0,0])

        color_palette = ["#5833ff","#da8829"]
        mask = np.triu(np.ones_like(df_corr))
        ax0.text(1.5,-0.1,"Correlation Matrix",fontsize=22, fontweight='bold', fontfamily='serif', color="#000000")
        df_corr = df_corr[self.con_cols].corr().transpose()
        sns.heatmap(df_corr, mask=mask, fmt=".1f", annot=True, cmap='YlGnBu')

        plt.show()

        fig = plt.figure(figsize=(12,12))
        corr_mat = self.data.corr().stack().reset_index(name="correlation")
        g = sns.relplot(
            data=corr_mat,
            x="level_0", y="level_1", hue="correlation", size="correlation",
            palette="YlGnBu", hue_norm=(-1, 1), edgecolor=".7",
            height=10, sizes=(50, 250), size_norm=(-.2, .8),
        )
        g.set(xlabel="features on X", ylabel="featurs on Y", aspect="equal")
        g.fig.suptitle('Scatterplot heatmap',fontsize=22, fontweight='bold', fontfamily='serif', color="#000000")
        g.despine(left=True, bottom=True)
        g.ax.margins(.02)
        for label in g.ax.get_xticklabels():
            label.set_rotation(90)
        for artist in g.legend.legendHandles:
            artist.set_edgecolor(".7")
        plt.show()

    # def corr_plot2(self):
    #     px.imshow(self.data.corr())

    def linearality(self):
        plt.figure(figsize=(18,18))
        for i, col in enumerate(self.data.columns, 1):
            plt.subplot(4, 3, i)
            sns.histplot(self.data[col], kde=True)
            plt.tight_layout()
            plt.plot()
        plt.show()


    def pairplot(self):
        sns.pairplot(self.data, hue=self.target, palette=["#8000ff","#da8829"])
        plt.show()
        sns.pairplot(self.data, hue=self.target, kind='kde')
        plt.show()

    def kde_plot(self):
        fig = plt.figure(figsize=(18,18))
        gs = fig.add_gridspec(1,2)
        gs.update(wspace=0.5, hspace=0.5)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[1])
        bg = "#ffe6e6"
        ax0.set_facecolor(bg) 
        ax1.set_facecolor(bg) 

        fig.patch.set_facecolor(bg)
        #sns.kdeplot(ax=ax0, data=self.data, x=self.feature, hue=self.target, zorder=0, dashes=(1,5))
        ax0.text(0.5, 0.5, "Distribution of " + str(self.feature) + " to\n " + str(self.target) + "\n",
            horizontalalignment = 'center',
            verticalalignment = 'center',
            fontsize = 18,
            fontweight='bold',
            fontfamily='serif',
            color='#000000')

        ax1.text(1, 0.25, "feature",
            horizontalalignment = 'center',
            verticalalignment = 'center',
            fontsize = 14
            )
        ax0.spines["bottom"].set_visible(False)
        ax0.set_xticklabels([])
        ax0.set_yticklabels([])
        ax0.tick_params(left=False, bottom=False)

        ax1.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
        sns.kdeplot(ax=ax1, data=self.data, x=self.feature, hue=self.target, alpha=0.7, linewidth=1, fill=True, palette=["#8000ff","#da8829"])
        ax1.set_xlabel("")
        ax1.set_ylabel("")

        for i in ["top","left","right"]:
            ax0.spines[i].set_visible(False)
            ax1.spines[i].set_visible(False)
        #sns.kdeplot(data=self.data, x=self.feature, hue=self.target, dashes=(1,5), alpha=0.7, linewidth=0, palette=["#8000ff","#da8829"])
        plt.show()

    def univariate_analysis(self, output_plot=None):

        if output_plot == 'output':
            self.target_plot()
        elif output_plot == 'corr':
            self.corr_plot()
        elif output_plot == 'pair':
            self.pairplot()
        elif output_plot == 'kde':
            self.kde_plot()
        elif output_plot == 'linearality':
            self.linearality()

if __name__ == "__main__":
        # Initialise the meta class
    meta_obj = ML_meta(data, all=False, model="MLP")
    meta_obj.apply_single_model()
    

