import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("car data.csv")

# Display basic info
print(df.info())
print(df.describe())

# Drop Car_Name column since it's not a useful numerical feature
df.drop(columns=['Car_Name'], inplace=True)

# Encode categorical features
label_encoders = {}
for column in ['Fuel_Type', 'Selling_type', 'Transmission']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Feature selection
X = df.drop(columns=['Selling_Price'])  # 'Selling_Price' is the target variable
y = df['Selling_Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}")

# Save model
import joblib

joblib.dump(model, "car_price_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")


# Function for new car price prediction
def predict_car_price(features):
    df_input = pd.DataFrame([features])
    for col, le in label_encoders.items():
        if col in df_input.columns:
            df_input[col] = le.transform(df_input[col])
    df_input = scaler.transform(df_input)
    return model.predict(df_input)[0]


# Example usage
sample_features = {'Year': 2015, 'Present_Price': 7.5, 'Driven_kms': 40000, 'Fuel_Type': 'Petrol',
                   'Selling_type': 'Dealer', 'Transmission': 'Manual', 'Owner': 0} 
predicted_price = predict_car_price(sample_features)
print(f"Predicted Price: {predicted_price}")
