import os
import sys
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import customException
from src.logger import logging
from src.utils import Haversine_distance,get_part_of_day
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts','data.csv')
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Starting Data Ingestion')
        try:
            df=pd.read_csv(r'Notebook\data\amazon_delivery.csv')
            logging.info('Read the Dataset')
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Starting Feature Engineering...")
            
            # === 2. FEATURE ENGINEERING (from your notebook) ===
            
            # Calculate Haversine distance
            df['Distance_km'] = df.apply(
                lambda row: Haversine_distance(
                    row['Store_Latitude'], row['Store_Longitude'],
                    row['Drop_Latitude'], row['Drop_Longitude']
                ), axis=1
            )
           
            # Handle missing values
            df.dropna(inplace=True)
           
            # Process Order date
            df['Order_Date'] = pd.to_datetime(df['Order_Date'])
            df['Order_Year'] = df['Order_Date'].dt.year
            df['Order_Month'] = df['Order_Date'].dt.month
            df['Order_Day'] = df['Order_Date'].dt.day
            df['day_of_week'] = df['Order_Date'].dt.day_name()
           
            #Process Order time
            df['Order_Time'] = pd.to_datetime(df['Order_Time'], format='%H:%M:%S')
            df['Order_Hour'] = df['Order_Time'].dt.hour
            df['Order_Minute'] = df['Order_Time'].dt.minute
            
            # Create 'order_part_of_day'
            df['part_of_day'] = df['Order_Hour'].apply(get_part_of_day)
            
            # Process Pickup_Time
            df['Pickup_Time'] = pd.to_datetime(df['Pickup_Time'], format='%H:%M:%S')
            df['Pickup_Hour'] = df['Pickup_Time'].dt.hour
            df['Pickup_Minute'] = df['Pickup_Time'].dt.minute
           
            # Calculate Total_preparation_time
            order_total_hours = df['Order_Hour'] + (df['Order_Minute'] / 60)
            pickup_total_hour = df['Pickup_Hour'] + (df['Pickup_Minute'] / 60)
            df['Total_preparation_time'] = pickup_total_hour - order_total_hours
            df.loc[df['Total_preparation_time'] < 0, 'Total_preparation_time'] += 24
            
            # Process Delivery_Time (This is our target variable)
            df['Delivery_Time_hour'] = (df['Delivery_Time'] / 60)

            # Drop unnecessary columns
            df.drop(columns=['Order_ID', 'Store_Latitude', 'Store_Longitude', 'Drop_Latitude', 'Drop_Longitude', 'Order_Date', 'Order_Time', 'Pickup_Time','Delivery_Time'], inplace=True)
            logging.info("Feature Engineering and cleaning complete")

           # === 3. TRAIN TEST SPLIT ===
            logging.info("Splitting data into train and test sets")
            train_set,test_set=train_test_split(df,test_size=0.3,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Ingestion of the data is complete')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            logging.error(f"Error in data ingestion: {e}")
            raise customException(e)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_training(train_arr,test_arr)) 


        

        
        
        
           
        






       