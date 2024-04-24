import os
import sys
import pickle
from src.exception import CustomException
from sklearn.metrics import accuracy_score
from src.logger import logging

def save_function(file_path, obj): 
    """Save object to file."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj: 
            pickle.dump(obj, file_obj)
    except Exception as e: 
        logging.info("Error occurred while saving object to file.")
        raise CustomException(e, sys)

def model_performance(X_train, y_train, X_test, y_test, models): 
    """Evaluate model performance using accuracy score."""
    try: 
        report = {}
        for model_name, model in models.items(): 
            # Train models
            model.fit(X_train, y_train)
            # Test data
            y_test_pred = model.predict(X_test)
            # Accuracy Score 
            test_model_score = accuracy_score(y_test, y_test_pred)
            report[model_name] = test_model_score
        return report
    except Exception as e: 
        logging.info("Error occurred while evaluating model performance.")
        raise CustomException(e, sys)

def load_obj(file_path):
    """Load object from file."""
    try: 
        with open(file_path, 'rb') as file_obj: 
            return pickle.load(file_obj)
    except Exception as e: 
        logging.info("Error occurred while loading object from file.")
        raise CustomException(e, sys)
