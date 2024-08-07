# Import all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from importlib import import_module

#Import all necessary libraries for CNN training
try:
    import tensorflow as tf
    from keras import layers
    from keras.models import Sequential
    from keras.layers import Input
    from keras.layers import Dense, Dropout, Lambda, AveragePooling2D, Flatten, Rescaling
    from keras.layers import Rescaling, RandomContrast, RandomZoom, RandomTranslation, RandomBrightness, RandomRotation
    from keras.layers import RandomFlip, RandomCrop
    from keras.losses import SparseCategoricalCrossentropy
    from keras.utils import image_dataset_from_directory    
    from keras.callbacks import EarlyStopping
    from keras.models import load_model
    from keras.optimizers import Adam
    from keras.applications import mobilenet_v2
    from keras.applications import MobileNetV2
    from keras.utils import image_dataset_from_directory
    from keras.utils import img_to_array
    #from tensorflow.keras.preprocessing.image import ImageDataGenerator
except ImportError:
    print("Unable to Import Tensorflow/Keras inside of the Base Classes script")
    exit(0)

from abc import ABC, abstractmethod
import inspect

# Optimizer retains its American spelling because that is the argument name for the methods required - I'm not happy about it either

def prepare_images(max_pixel_val, width, height, data_path, batch_size):
    """
    A function to prepare images for use in a Convolutional Neural Network (CNN).

    This function uses the ImageDataGenerator from keras to 
    """

    trainDs = image_dataset_from_directory(directory=data_path, 
                                            validation_split=0.2,
                                            subset="training",
                                            seed=123,
                                            image_size=(height, width),
                                            batch_size=batch_size)
    valDs = image_dataset_from_directory(directory=data_path, 
                                            validation_split=0.2,
                                            subset="validation",
                                            seed=123,
                                            image_size=(height, width),
                                            batch_size=batch_size)
    #imageGen = ImageDataGenerator(rescale=1/max_pixel_val, validation_split=0.2)
    #trainDatagen = imageGen.flow_from_directory(directory=data_path, target_size=(width,height), class_mode='binary',
    #                                            batch_size=16, subset='training')
    #valDatagen = imageGen.flow_from_directory(directory=data_path, target_size=(width,height), class_mode='binary',
    #                                            batch_size=16, subset='validation')
    
    return trainDs, valDs

class CNN_config():
    """
     A class for configuring and managing Convolutional Neural Networks (CNN).

    This class provides functionality to read CNN configurations from a file,
    create CNN models based on those configurations, compile and train the models,
    and visualise feature maps.

    Attributes:
        path (str): Path to the configuration file.
        optimizer: The optimizer to be used for model compilation.
        loss: The loss function to be used for model compilation.
        metrics: The metrics to be used for model evaluation.
        train_images: Training image data.
        test_images: Test image data.
        train_labels: Training labels.
        test_labels: Test labels.
        config_list (list): List to store configuration parameters.

    Methods:
        read(): Reads and processes the configuration file.
        createCNN(): Creates a CNN model based on the configuration.
        model_summary(): Displays a summary of the created model.
        model_create(): Compiles and trains the CNN model.
        feature_map(model, image): Generates and displays feature maps for a given image.
    """
    def __init__(self, path, optimizer, loss, metrics, train_images=None, test_images=None, train_labels=None, test_labels=None, config_list=[], conv_iter=0, dense_iter=0) -> None:
        """
        Initialises the CCN_config class with the given parameters.

        Args:
            path (str): Path to the configuration file.
            optimizer: The optimizer to be used for model compilation.
            loss: The loss function to be used for model compilation.
            metrics: The metrics to be used for model evaluation.
            train_images: Training image data.
            test_images: Test image data.
            train_labels: Training labels.
            test_labels: Test labels.
            config_list (list, optional): Initial configuration list. Defaults to an empty list.
        """
        self.path = path
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.train_images = train_images
        self.test_images = test_images
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.config_list = config_list
        self.conv_iter = conv_iter
        self.dense_iter = dense_iter

    def read(self):
        """
        Reads the configuration file and processes its contents.

        Returns:
            list: A list of processed configuration parameters.
        """
        config_list = []
        with open(self.path, 'r') as data:
            for line in data:
                config_list.append(line)
        config_list = [(x.replace('\n', '')) for x in config_list]
        config_list = [(x.replace(' ', '')) for x in config_list]

        return config_list
    
    def createCNN(self):
        """
        Creates a CNN model based on the configuration read from the file.

        Returns:
            tensorflow.keras.Model: The created CNN model.
        """
        config_list = self.read()
        model = None
        
        for item in range(len(config_list)):
            if item == 0:
                if "Sequential" in config_list[item]:
                    model = Sequential()
                else:
                    print("Model must start with a sequential layer")
                    return None  # or raise an exception

            else:
                if model is None:
                    print("Model was not initialized properly")
                    return None  # or raise an exception

                if "Rescaling" in config_list[item]:
                    model.add(layers.Rescaling(1./255))

                elif "Conv2D" in config_list[item]:
                    item_splits = config_list[item].split(",")
                    if self.conv_iter < 1:
                        model.add(layers.Conv2D(int(item_splits[1]), (int(item_splits[2]), int(item_splits[3])), 
                                                activation=item_splits[4], input_shape=(int(item_splits[1]), int(item_splits[1]), int(item_splits[2]))))
                        self.conv_iter += 1
                    else:
                        model.add(layers.Conv2D(int(item_splits[1]), (int(item_splits[2]), int(item_splits[3])), 
                                                activation=item_splits[4]))
                    
                elif "MaxPooling2D" in config_list[item]:
                    item_splits = config_list[item].split(",")
                    model.add(layers.MaxPooling2D((int(item_splits[1]), int(item_splits[2]))))

                elif "Flatten" in config_list[item]:
                    model.add(layers.Flatten())

                elif "Dense" in config_list[item]:
                    item_splits = config_list[item].split(",")
                    if self.dense_iter < 0:
                        model.add(layers.Dense(int(item_splits[1]), activation=item_splits[2]))
                        print(item_splits)
                    else:
                        model.add(layers.Dense(int(item_splits[1])))
        
        return model


    def model_summary(self):
        """
        Displays a summary of the created CNN model.

        Returns:
            model summary
        """
        
        model = self.model_create()

        model_summary = model.summary()
        
        return model_summary

    def model_create(self):
        """
        Compiles and trains the CNN model.

        Returns:
            tuple: A tuple containing the compiled model and its training history.
        """
        
        model = self.createCNN()

        model.compile(optimizer=self.optimizer,
                    loss = tf.keras.losses.SparseCategoricalCrossEntropy(from_logits=True),
                    metrics=[self.metrics])

        history = model.fit(self.train_images, self.train_labels, epochs=10,
                            validation_data=(self.test_images, self.test_labels))

        return model, history
    
    def feature_map(self, model, image):
        """
        Generates and displays feature maps for a given image using the trained model.

        Args:
            model: The trained CNN model.
            image (str): Path to the input image file.

        Returns:
            No return
            Plot displayed
        """
        model = self.model_create()

        #img = load_img(image, target_size=(224, 224))   
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        #img = preprocess_input(img) # Need to use a different function for this

        feature_maps = model.predict(img)

        square = 8
        ix = 1

        for _ in range(square):
            for _ in range(square):
                ax = plt.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')

                ix += 1
        plt.show()

    def show_performance(self, history, metric, metric_label):
        """
        Generates plots to compare the performance of the model on both the training and test sets.

        Args:
            history: The trained model 
            metric: A measure of the performance of the model (string)

        Returns:
            No return
            Plot displayed
        """
        if (isinstance(metric, str)):
            train_performance = history.history[metric]
            valid_performance = history.history['val_' + metric]
            intersection_index = np.argwhere(np.isclose(train_performance, valid_performance, atol=1e-2)).flatten()[0]
            intersecion_val = train_performance[intersection_index]
        else:
            print("metric must be of string type")
            exit(0)


        plt.plot(train_performance, label=metric_label)
        plt.plot(valid_performance, label='val_'+metric)

        plt.axvline(x=intersection_index, color='r', linestyle='--', label='Intersecion Index')
        plt.annotate(f'Optimal Value: {intersecion_val}:.4f', xy=(intersection_index, intersecion_val),
                     xycoords='data', fontsize=12, color='g')
        
        plt.xlabel('Epoch')
        plt.ylabel(metric_label)
        plt.legend(loc='lower right')

    def cm_plot(self, model):
        """
        A method to plot the confusion matrix of an input trained CNN model.

        Args:
            model: The trained CNN model.

        Returns:
            No return.
            Plots confusion matrix to display
        """
        try:
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        except ImportError("Error importing skearn methods"):
            exit(0)

        test_pred = model.predict(self.test_images)
        test_pred_labels = np.argmax(test_pred, axis=1)
        test_truth_labls = np.argmax(self.test_labels, axis=1)

        cm = confusion_matrix(test_truth_labls, test_pred_labels)
        cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        cm_disp.plot(include_values=True, cmap="viridis", ax=None, xticks_rotation='horizontal')
        plt.show()

        

if __name__ == "__main__":
    CNN_build = CNN_config(r"C:\Users\chris\OneDrive\Documents\HeimPy\src\HeimdallX\CNN_build.txt", "Adam", None, 'accuracy')

    CNN_build.createCNN()