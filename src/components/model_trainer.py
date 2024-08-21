import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.logger import logging
from src.components.model_creation import CreateModel
from src.exception import CustomException
import h5py

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.keras')


class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def generator(self, data, label, batch_size):
        num_batch = data.shape[0] // batch_size
        
        while True:
            start = 0
            for batch in range(num_batch):
                batch_data = np.zeros((batch_size, 28, 28, 1))
                batch_label = np.zeros((batch_size, 10), dtype='int')
                for i in range(batch_size):
                    batch_data[i,:,:,:] = data[start+i,:,:]
                    batch_label[i,label[start+i,0]] = 1
                start += batch_size
                yield batch_data, batch_label

    def initiate_model_training(self, X_train, y_train, X_val, y_val):
        try:
            logging.info("Initiating model training")
            batch_size = 100
            train_generator = self.generator(X_train, y_train, batch_size)
            val_generator = self.generator(X_val, y_val, batch_size)

            model = CreateModel().createCNNModel()

            steps_per_epoch = X_train.shape[0] // batch_size
            validation_steps = X_val.shape[0] // batch_size

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

            model.fit(train_generator, epochs=1, verbose=1, validation_data=val_generator, 
                      steps_per_epoch=steps_per_epoch, validation_steps=validation_steps
            )
            logging.info("Model has been fitted")
            categoriacl_accuracy = model.get_metrics_result()['categorical_accuracy'].numpy()
            print("Validation Categorical Accuracy is ", categoriacl_accuracy)
            
            model.save(self.model_trainer_config.trained_model_file_path)

        except Exception as e:
            raise CustomException(e, sys)
