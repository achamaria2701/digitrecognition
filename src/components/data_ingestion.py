import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    val_data_path: str = os.path.join('artifacts', 'val.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        raw_train_path = os.path.join('notebook','data','train.csv')
        try:
            df_train = pd.read_csv(raw_train_path)

            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df_train.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            train_set, val_set = train_test_split(df_train, train_size=0.8, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            val_set.to_csv(self.ingestion_config.val_data_path, index=False, header=True)

            logging.info("Ingestion of data completed")

            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.val_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train, val, = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    X_tr, y_tr, X_v, y_v = data_transformation.initiate_data_transformation(train, val)
    ModelTrainer().initiate_model_training(X_tr, y_tr, X_v, y_v)