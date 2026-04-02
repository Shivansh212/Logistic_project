import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import load_object # Using your custom tool!



# 1. Load your test data and your saved objects
test_df = pd.read_csv('artifacts/test.csv')
model = load_object('artifacts/best_model.pkl')
preprocessor = load_object('artifacts/preprocessor.pkl')

# 2. Separate features and the target variable
X_test = test_df.drop(columns=['Delivery_Time_hour'])
y_test = test_df['Delivery_Time_hour']

# 3. Transform the data and make predictions
X_test_scaled = preprocessor.transform(X_test)
predictions = model.predict(X_test_scaled)

# 4. Calculate the Residuals (The Errors)
residuals = y_test - predictions

# 5. Plot the graph
plt.figure(figsize=(10, 6))
sns.scatterplot(x=predictions, y=residuals, alpha=0.5, color='teal')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2) # The "Perfect Prediction" line
plt.title('Residual Plot: Model Error Analysis')
plt.xlabel('Predicted Delivery Time (Hours)')
plt.ylabel('Error (Actual Time - Predicted Time)')
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# 1. Calculate the core metrics (using the 'y_test' and 'predictions' from earlier)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

# 2. Print them out nicely formatted in Hours and Minutes
print("-" * 40)
print("MODEL EVALUATION METRICS")
print("-" * 40)
print(f"MAE (Mean Absolute Error):    {mae:.4f} hours  (~ {mae * 60:.1f} minutes)")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f} hours  (~ {rmse * 60:.1f} minutes)")
print(f"MSE (Mean Squared Error):     {mse:.4f}")
print("-" * 40)