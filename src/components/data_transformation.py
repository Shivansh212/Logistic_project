import os
import sys

import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.exception import customException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns=['Agent_Age','Agent_Rating','Distance_km','Order_Year','Order_Month','Order_Day','Order_Hour','Order_Minute','Pickup_Hour','Pickup_Minute','Total_preparation_time']
            categorical_columns=['Weather','Traffic','Vehicle','Area','Category','day_of_week','part_of_day']

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            num_pipeline=Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", RobustScaler())                  
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")), 
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')), 
                    ("scaler", RobustScaler(with_centering=False)) 
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns), # Apply num_pipeline to numerical_columns
                    ("cat_pipeline", cat_pipeline, categorical_columns)  # Apply cat_pipeline to categorical_columns
                ]
            )

            logging.info("Preprocessor ColumnTransformer object created.")
            
            return preprocessor

        except Exception as e:
            raise customException(e)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed.")
            logging.info("Obtaining preprocessing object...")

            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "Delivery_Time_hour"
            #Training data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            #Testing data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info('Saving preprocessing object')

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise customException(e)
        


                                  
