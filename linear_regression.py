# Python script for Linear Regression to predict house prices

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Generate a synthetic dataset
np.random.seed(42)  # For reproducibility
n_samples = 500

# Generate features
square_feet = np.random.randint(500, 4000, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)

# Generate prices with some realistic relationships
price = (
    square_feet * 150  # Price increases with square footage
    + bedrooms * 10000  # Price increases with number of bedrooms
    + bathrooms * 5000  # Price increases with number of bathrooms
    + np.random.randint(-20000, 20000, n_samples)  # Add some noise
)

# Create a DataFrame
data = pd.DataFrame({
    'square_feet': square_feet,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'price': price
})

# Step 2: Data preprocessing
# Check for missing values
if data.isnull().sum().any():
    data = data.fillna(data.mean())  # Fill missing values with column mean

# Split the data into features (X) and target (y)
X = data[['square_feet', 'bedrooms', 'bathrooms']]
y = data['price']

# Split into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Implement Linear Regression
# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Step 4: Evaluate the model
# Make predictions
y_pred = model.predict(X_test)

# Calculate R² score and Mean Squared Error (MSE)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Print model performance
print("Model Performance:")
print(f"R² Score: {r2:.2f}")
print(f"Mean Squared Error: {mse:.2f}")

# Step 5: Visualizations
# 1. Actual vs. Predicted Prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', linewidth=2)
plt.title('Actual vs. Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.grid()
plt.show()

# 2. Feature Coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

plt.figure(figsize=(8, 5))
plt.bar(coefficients['Feature'], coefficients['Coefficient'], color='green')
plt.title('Feature Importance (Coefficients)')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.grid(axis='y')
plt.show()

# End of script