import sys
import os
import pandas as pd
from src.exception import customException
from src.logger import logging
from src.utils import load_object, Haversine_distance, get_part_of_day

class PredictPipeline:
    def __init__(self):
        # We don't need to load models here,
        # we can load them when predict() is called
        # to save memory if the server is just 'on' but not predicting.
        pass

    def predict(self, features):
        '''
        This function takes the 'features' (from the CustomData class)
        and runs them through the entire pipeline.
        '''
        try:
            # === 1. DEFINE FILE PATHS ===
            # This is the "address" to our saved model and preprocessor
            model_path = os.path.join('artifacts', 'best_model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            logging.info("Loading saved model and preprocessor...")
            
            # === 2. LOAD SAVED OBJECTS ===
            # We use our "Package Opener" tool from utils.py
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            logging.info("Loaded objects successfully.")

            # === 3. RE-APPLY FEATURE ENGINEERING ===
            # This is CRITICAL. We must do the *exact same* steps
            # as in data_ingestion.py to the new raw data.
            
            # 'features' is the DataFrame we get from our CustomData class
            
            # Calculate Haversine distance
            features['Distance_km'] = features.apply(
                lambda row: Haversine_distance(
                    row['Store_Latitude'], row['Store_Longitude'],
                    row['Drop_Latitude'], row['Drop_Longitude']
                ), axis=1
            )
            
            # Process Order_Date
            features['Order_Date'] = pd.to_datetime(features['Order_Date'])
            features['Order_Year'] = features['Order_Date'].dt.year
            features['Order_Month'] = features['Order_Date'].dt.month
            features['Order_Day'] = features['Order_Date'].dt.day
            features['day_of_week'] = features['Order_Date'].dt.day_name()
            
            # Process Order_Time
            features['Order_Time'] = pd.to_datetime(features['Order_Time'], format='%H:%M:%S')
            features['Order_Hour'] = features['Order_Time'].dt.hour
            features['Order_Minute'] = features['Order_Time'].dt.minute
            
            # Create 'order_part_of_day'
            features['part_of_day'] = features['Order_Hour'].apply(get_part_of_day)
            
            # Process Pickup_Time
            features['Pickup_Time'] = pd.to_datetime(features['Pickup_Time'], format='%H:%M:%S')
            features['Pickup_Hour'] = features['Pickup_Time'].dt.hour
            features['Pickup_Minute'] = features['Pickup_Time'].dt.minute
            
            # Calculate Total_preparation_time
            order_total_hours = features['Order_Hour'] + (features['Order_Minute'] / 60)
            pickup_total_hour = features['Pickup_Hour'] + (features['Pickup_Minute'] / 60)
            features['Total_preparation_time'] = pickup_total_hour - order_total_hours
            features.loc[features['Total_preparation_time'] < 0, 'Total_preparation_time'] += 24
            
            # Drop columns we don't need for the model
            # We keep the ones that the preprocessor is expecting!
            features_to_keep = [
                'Agent_Age','Agent_Rating','Weather','Traffic','Vehicle','Area','Category','Distance_km','Order_Year','Order_Month','Order_Day','day_of_week','Order_Hour','Order_Minute','part_of_day','Pickup_Hour','Pickup_Minute','Total_preparation_time','Delivery_Time_hour'
            ]
            
            final_features_df = features[features_to_keep]
            
            logging.info("Feature Engineering complete for new data.")
            logging.info(f"Final columns for preprocessing: {final_features_df.columns.to_list()}")

            # === 4. TRANSFORM & PREDICT ===
            # Use the loaded preprocessor to transform the new data
            data_scaled = preprocessor.transform(final_features_df)
            
            # Use the loaded model to make a prediction
            prediction = model.predict(data_scaled)
            
            logging.info("Prediction complete.")
            
            return prediction[0] # Return the single predicted value

        except Exception as e:
            logging.error(f"Error in prediction pipeline: {e}")
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
                 Total_preparation_time: float,
                 Delivery_Time_hour: float): 
        
        # This just saves all the inputs as attributes
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
        self.Delivery_Time_hour = Delivery_Time_hour

    def get_data_as_data_frame(self):
        
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
                "Total_preparation_time": [self.Total_preparation_time],
                "Delivery_Time_hour": [self.Delivery_Time_hour]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise customException(e)