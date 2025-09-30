import sys
import os
from source.components.data_ingestion import DataIngestion
from source.components.data_transformation import DataTransformation
from source.components.model_trainer import ModelTrainer
from source.exception import customException
from source.logger import logging

class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("Starting the training pipeline")

            # Data Ingestion
            data_ingestion = DataIngestion()
            train_path, test_path = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion completed")

            # Data Transformation
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_path, test_path)
            logging.info("Data transformation completed")

            # Model Training
            model_trainer = ModelTrainer()
            model_trainer.initiate_model_trainer(train_arr, test_arr)
            logging.info("Model training completed")

            logging.info("Training pipeline completed successfully")

        except Exception as e:
            raise customException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
