import os
import sys
import pandas as pd
from dataclasses import dataclass


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import customException
from src.logger import logging

from src.utils import evaluate_model, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "best_model.pkl")

class ModelTrainer:
    def __init__(self):
        
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        
        
        try:
            logging.info("Starting model training component...")
            logging.info("Splitting training and test data into X and y")
            
            
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
            }
            
            params = {
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error'],
                },
                "Random Forest":{
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "Lasso": {},
                "Ridge": {},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoost Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }
            
            logging.info("Evaluating all models...")
            
            # === 3. RUN THE CONTEST ===
            # We use our "Contest Judge" tool from utils.py
            # It will train and test every model in our dictionary
            # and return a scorecard.
            model_report, fitted_models = evaluate_model(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models,
                param_grid=params  # <-- You might have added this
            )
            
            # === 4. FIND THE WINNER ===
            # Get the best model's name and score from the report
            
            # Get all the scores (R2 score) from the report
            all_scores = [model['r2_score'] for model in model_report.values()]
            best_model_score = max(all_scores)
            
            # Get the name of the model that had the best score
            best_model_name = list(model_report.keys())[
                all_scores.index(best_model_score)
            ]
            
            # Get the actual model object (the "winner")
            best_model = fitted_models[best_model_name]

            # Set a minimum score. We don't want to save a bad model.
            if best_model_score < 0.6:
                logging.warning(f"No model performed well. Best R2 score: {best_model_score}")
                raise customException("No model reached the minimum R2 score of 0.6")

            logging.info(f"Best model found:")
            logging.info(f"Model Name: {best_model_name}")
            logging.info(f"Model R2 Score: {best_model_score}")

            # === 5. SAVE THE WINNER ===
            # Use our "Vacuum Sealer" tool from utils.py
            # to save the winning model to the 'artifacts' folder.
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # === 6. RETURN THE SCORE ===
            # Return the winner's score so we know how well it did.
            return best_model_score

        except Exception as e:
            logging.error(f"Error in model training: {e}")
            raise customException(e)