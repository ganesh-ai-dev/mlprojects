import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from source.exception import customException
from source.logger import logging
from source.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            model = LinearRegression()
            model.fit(X_train, y_train)
            logging.info("Model training completed")

            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            logging.info(f"R2 score on test data: {r2}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
            logging.info("Model saved successfully")

            return r2

        except Exception as e:
            raise customException(e, sys)
