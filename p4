# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score\

# Step 2: Load the dataset
df = pd.read_csv("BostonHousing.csv")
# Step 3: Check the dataset
print("Dataset shape:", df.shape)
print(df.head())

X = df.drop(columns='medv')
y = df['medv']

# Step 5: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Create and train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
print("\nModel Coefficients:")
print(model.coef_)
print("\nIntercept:", model.intercept_)

# Model performance
print("\nMean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R-squared (R2 Score):", r2_score(y_test, y_pred))

#Step 9: Plot predictions vs actual
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.grid(True)
plt.show()
