import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import os

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def transform_data(self, data):
        try:
            logging.info("Entered transform data method")
            tot_samples = data.shape[0]
            y = data.pop('label')
            X = data
            y_np = np.array(y).reshape((-1,1))
            X_np = np.array(X).reshape((tot_samples, 28, 28, 1))
            return X_np, y_np
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, val_path):
        try:
            logging.info("In initiate data transformation method")
            
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            
            logging.info("Files reading completed as dataframe")

            X_train, y_train = self.transform_data(train_df)
            X_val, y_val = self.transform_data(val_df)
        
            logging.info("Files converted to numpy array")

            return (X_train, y_train,
                    X_val, y_val
            )
        except Exception as e:
            raise CustomException(e, sys)