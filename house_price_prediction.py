# House Price Prediction using Linear Regression
# Author: Cherukuru Hemalatha

# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# Step 1: Load Dataset
# -------------------------------

# Load the Kaggle house price dataset
data = pd.read_csv("train.csv")

# Display first few rows
print("Dataset Preview:")
print(data.head())

# -------------------------------
# Step 2: Select Important Features
# -------------------------------

# Features used for prediction
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']

# Independent variables
X = data[features]

# Target variable
y = data['SalePrice']

# -------------------------------
# Step 3: Handle Missing Values
# -------------------------------

# Fill missing values with mean
X = X.fillna(X.mean())

# -------------------------------
# Step 4: Split Dataset
# -------------------------------

# 80% training data
# 20% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Step 5: Train Model
# -------------------------------

# Create Linear Regression model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# -------------------------------
# Step 6: Predict House Price
# -------------------------------

# Example new house
new_house = pd.DataFrame([[2000, 3, 2]],
                         columns=features)

predicted_price = model.predict(new_house)

print("\nPredicted House Price:", predicted_price[0])

# -------------------------------
# Step 7: Evaluate Model
# -------------------------------

# Predict test data
y_pred = model.predict(X_test)

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# R2 Score
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
