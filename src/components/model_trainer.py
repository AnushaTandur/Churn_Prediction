import sys
from sklearn.linear_model import LogisticRegression
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import os

@dataclass 
class ModelTrainerConfig():
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
       

    def initiate_model_training(self, X_train, y_train):
        try:
            logging.info("Training Logistic Regression model")
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            logging.info("Logistic Regression model training completed")
            return model
        except Exception as e:
            logging.error("Error occurred during model training")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        # Example usage
        model_trainer = ModelTrainer()
        X_train = [...]  # Your training features
        y_train = [...]  # Your training labels
        trained_model = model_trainer.initiate_model_training(X_train, y_train)
    except CustomException as ce:
        logging.error(f"CustomException occurred: {ce}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
