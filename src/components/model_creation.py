import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import models, layers

from src.logger import logging
from src.exception import CustomException


class CreateModel():    
    def createCNNModel(self):
        try:
            logging.info("Inside createCNNModel function")
            model = models.Sequential()
            model.add(layers.Rescaling(scale=1./255, input_shape=(28,28,1)))

            model.add(layers.Conv2D(32, kernel_size=3, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Conv2D(32, kernel_size=2, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Conv2D(32, kernel_size=2, strides=2, padding='same', activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(0.4))

            model.add(layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(0.4))

            model.add(layers.Conv2D(128, kernel_size=4, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Flatten())
            model.add(layers.Dropout(0.4))
            model.add(layers.Dense(10, activation='softmax'))
            return model
        
        except Exception as e:
            raise CustomException(e, sys)

