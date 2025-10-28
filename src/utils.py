import os
import sys
import dill

from src.logger import logging
from src.exception import customException

from math import radians,sin,cos,atan2,sqrt
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV

def Haversine_distance(lat1,long1,lat2,long2):
    R=6371
    dlat = radians(lat2 - lat1)
    dlon = radians(long2 - long1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

def get_part_of_day(hour):

    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'
    

def evaluate_model(X_train, y_train, X_test, y_test, models, param_grid):
    
    try:
        report = {}
        best_models = {}  
        logging.info("Starting model evaluation with GridSearchCV...")

        for model_name, model in models.items():
            logging.info(f"Tuning model: {model_name}")
            
            
            model_params = param_grid.get(model_name, {})

            
            gs = GridSearchCV(
                estimator=model,
                param_grid=model_params,
                cv=3,      
                 
                scoring='r2'
            )

            
            gs.fit(X_train, y_train)

            
            best_model_from_grid = gs.best_estimator_
            
            # Store this best model
            best_models[model_name] = best_model_from_grid
            
            # --- Now, evaluate this best model ---
            y_test_pred = best_model_from_grid.predict(X_test)
            test_model_r2 = r2_score(y_test, y_test_pred)
            test_model_mae = mean_absolute_error(y_test, y_test_pred)
            
            
            report[model_name] = {
                'r2_score': test_model_r2,
                'mae': test_model_mae,
                'best_params': gs.best_params_  
            }
            
            logging.info(f"Finished tuning {model_name}. Best R2: {test_model_r2:.4f}")

        return report, best_models  
        
    except Exception as e:
        raise customException(e)
    
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise customException(e)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
            
    except Exception as e:
        raise customException(e)