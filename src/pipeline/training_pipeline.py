# import sys
# from src.exception import customException
# from src.logger import logging


# from src.components.data_ingestion import DataIngestion
# from src.components.data_transformation import DataTransformation
# from src.components.model_trainer import ModelTrainer


# if __name__ == "__main__":
    
    
#     try:
#         logging.info("Starting the training pipeline...")

#         # === 2. TELL THE "Data Ingestion" TO START ===
#         logging.info("Running Data Ingestion...")
#         ingestion_obj = DataIngestion()
        
        
#         train_data_path, test_data_path = ingestion_obj.initiate_data_ingestion()
        
#         logging.info(f"Data ingestion complete. Train/Test files are at: {train_data_path}, {test_data_path}")

#         # === 2. TELL THE "Data Transformation" TO START ===
#         logging.info("Running Data Transformation...")
#         transformation_obj = DataTransformation()
        
        
#         train_arr, test_arr, preprocessor_file_path = transformation_obj.initiate_data_transformation(
#             train_path=train_data_path, 
#             test_path=test_data_path
#         )
        
#         logging.info(f"Data transformation complete. Preprocessor saved at: {preprocessor_file_path}")

#         # === 3. TELL THE "Model Trainer" TO START ===
#         logging.info("Running Model Training...")
#         trainer_obj = ModelTrainer()
        
        
#         best_r2_score = trainer_obj.initiate_model_training(
#             train_array=train_arr,
#             test_array=test_arr
#         )
        
#         logging.info(f"Model training complete. Best model R2 score: {best_r2_score}")
#         logging.info("Training pipeline finished successfully!")

#     except Exception as e:
        
#         logging.error("An error occurred in the main training pipeline.")
#         raise customException(e)