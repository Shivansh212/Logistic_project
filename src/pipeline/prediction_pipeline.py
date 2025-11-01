import sys
import os
import pandas as pd
from src.exception import customException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        '''
        This function takes the features DataFrame and runs prediction.
        '''
        try:
            # === 1. DEFINE FILE PATHS ===
            model_path = os.path.join('artifacts', 'best_model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            logging.info("Loading saved model and preprocessor...")
            
            # === 2. LOAD SAVED OBJECTS ===
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            logging.info("Loaded objects successfully.")

           

            logging.info("Feature Engineering skipped, features provided directly.")
            logging.info(f"Final columns for preprocessing: {features.columns.to_list()}")

            # === 4. TRANSFORM & PREDICT ===
            # Use the loaded preprocessor to transform the new data
            data_scaled = preprocessor.transform(features) 
            
            # Use the loaded model to make a prediction
            prediction = model.predict(data_scaled)
            
            return prediction[0] # Return the single predicted value

        except Exception as e:
            raise customException(e)
            


class CustomData:
    
    def __init__(self,
                 Agent_Age: float,
                 Agent_Rating: float,
                 Weather: str,
                 Traffic: str,
                 Vehicle: str,
                 Area: str,
                 Category: str,
                 Distance_km: float,
                 Order_Year: float,
                 Order_Month: float,
                 Order_Day: float,
                 day_of_week: str,
                 Order_Hour: float,
                 Order_Minute: float,
                 part_of_day: str,
                 Pickup_Hour: float, 
                 Pickup_Minute: float,
                 Total_preparation_time: float): 
        
        # This saves all the inputs as attributes
        self.Agent_Age = Agent_Age
        self.Agent_Rating = Agent_Rating
        self.Weather = Weather
        self.Traffic = Traffic
        self.Vehicle = Vehicle
        self.Area = Area
        self.Category = Category
        self.Distance_km = Distance_km
        self.Order_Year = Order_Year
        self.Order_Month = Order_Month
        self.Order_Day = Order_Day
        self.day_of_week = day_of_week
        self.Order_Hour = Order_Hour
        self.Order_Minute = Order_Minute
        self.part_of_day = part_of_day
        self.Pickup_Hour = Pickup_Hour
        self.Pickup_Minute = Pickup_Minute
        self.Total_preparation_time = Total_preparation_time

    def get_data_as_data_frame(self):
        '''
        This function takes all the raw inputs from the web form
        and converts them into a pandas DataFrame.
        This is the *input* for our PredictPipeline.
        '''
        try:
            custom_data_input_dict = {
                "Agent_Age": [self.Agent_Age],
                "Agent_Rating": [self.Agent_Rating],
                "Weather": [self.Weather],
                "Traffic": [self.Traffic],
                "Vehicle": [self.Vehicle],
                "Area": [self.Area],
                "Category": [self.Category],
                "Distance_km": [self.Distance_km],
                "Order_Year": [self.Order_Year],
                "Order_Month": [self.Order_Month],
                "Order_Day": [self.Order_Day],
                "day_of_week": [self.day_of_week],
                "Order_Hour": [self.Order_Hour],
                "Order_Minute": [self.Order_Minute],
                "part_of_day": [self.part_of_day],
                "Pickup_Hour": [self.Pickup_Hour],
                "Pickup_Minute": [self.Pickup_Minute],
                "Total_preparation_time": [self.Total_preparation_time]
                
            }

            return pd.DataFrame(custom_data_input_dict)
            

        except Exception as e:
            raise customException(e)




