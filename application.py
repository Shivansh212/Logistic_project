import numpy as np
import pandas as pd
from flask import Flask, request, render_template

# Import your custom classes from the prediction pipeline
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
from src.exception import customException
from src.logger import logging

application = Flask(__name__)
app = application

# Route for the main welcome page (index.html)
@app.route('/')
def index():
    return render_template('index.html')

# Route for the prediction form (home.html)
# This route handles both GET (showing the form) and POST (submitting the form)
@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        # Just show the form page
        return render_template('home.html')
    else:
        # This is a POST request, so we process the form data
        try:
            
            data = CustomData(
                Agent_Age=float(request.form.get('Agent_Age')),
                Agent_Rating=float(request.form.get('Agent_Rating')),
                Weather=str(request.form.get('Weather')),
                Traffic=str(request.form.get('Traffic')),
                Vehicle=str(request.form.get('Vehicle')),
                Area=str(request.form.get('Area')),
                Category=str(request.form.get('Category')),
                Distance_km=float(request.form.get('Distance_km')),
                Order_Year=float(request.form.get('Order_Year')),
                Order_Month=float(request.form.get('Order_Month')),
                Order_Day=float(request.form.get('Order_Day')),
                day_of_week=str(request.form.get('day_of_week')),
                Order_Hour=float(request.form.get('Order_Hour')),
                Order_Minute=float(request.form.get('Order_Minute')),
                part_of_day=str(request.form.get('part_of_day')),
                Pickup_Hour=float(request.form.get('Pickup_Hour')),
                Pickup_Minute=float(request.form.get('Pickup_Minute')),
                Total_preparation_time=float(request.form.get('Total_preparation_time'))
            )

            
            pred_df = data.get_data_as_data_frame()
            logging.info(f"New prediction input data: {pred_df.to_dict()}")
            
            
            predict_pipeline = PredictPipeline()
            prediction_hours = predict_pipeline.predict(pred_df)
            
            # Convert prediction to minutes
            prediction_minutes = round(prediction_hours * 60, 2)
            
            result_text = f"Estimated Delivery Time: {prediction_minutes} minutes"

            
            return render_template('result.html', prediction_text=result_text)

        except Exception as e:
            logging.error("Error occurred in /home POST route")
            raise customException(e)


if __name__ == "__main__":
    
    app.run(host="0.0.0.0", port=5002, debug=True)

