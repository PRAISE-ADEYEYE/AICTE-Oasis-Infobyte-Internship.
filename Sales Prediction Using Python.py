import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
file_path = "Advertising.csv"
df = pd.read_csv(file_path)

# Drop the unnamed index column
df = df.drop(columns=["Unnamed: 0"], errors='ignore')

# Define features and target variable
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

# Print evaluation metrics
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Model coefficients
coefficients = dict(zip(X.columns, model.coef_))
print("Feature Coefficients:", coefficients)

# Visualization of feature importance
plt.figure(figsize=(8, 5))
sns.barplot(x=pd.Series(coefficients.keys()), y=pd.Series(coefficients.values()), palette="viridis")
plt.title("Feature Importance in Sales Prediction")
plt.ylabel("Coefficient Value")
plt.xlabel("Advertising Channel")
plt.show()

# Scatter plots for each feature vs Sales
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.scatterplot(x=df["TV"], y=df["Sales"], ax=axes[0], color="blue")
axes[0].set_title("TV Advertising vs Sales")

sns.scatterplot(x=df["Radio"], y=df["Sales"], ax=axes[1], color="green")
axes[1].set_title("Radio Advertising vs Sales")

sns.scatterplot(x=df["Newspaper"], y=df["Sales"], ax=axes[2], color="red")
axes[2].set_title("Newspaper Advertising vs Sales")

plt.show()