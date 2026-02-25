# Imports
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split #flashcards?
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Loading Data
data = pd.read_csv("vehicle_emissions.csv")
data.head()
data.info()

# Create features (x) and target variables (y)
x = data.drop(["CO2_Emissions"], axis=1)
y = data["CO2_Emissions"]

# Split Categorial and Numerical Features
numerical_cols = ["Model_Year", "Engine_Size", "Cylinders", "Fuel_Consumption_in_City(L/100 km)","Fuel_Consumption_in_City_Hwy(L/100 km)","Fuel_Consumption_comb(L/100km)","Smog_Level"]
categorical_cols = ["Make","Model","Vehicle_Class","Transmission"]

# Combine the pipelines
numerical_pipeline = Pipeline ([
    ("imputer",SimpleImputer(strategy="mean")),
    ("scaler",StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("encoder",OneHotEncoder(handle_unknown="ignore"))
])

# Join pipelines together
preprocessor = ColumnTransformer([
    ('num',numerical_pipeline, numerical_cols),
    ('cat',categorical_pipeline, categorical_cols)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor())
])

# Split into training and testing
x_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

# Train and predict model
pipeline.fit(x_train, y_train)

prediction = pipeline.predict(X_test)

# View encoding
encoded_cols = pipeline.named_steps['preprocessor'].named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols)

#print(encoded_cols)

# Evaluate model accuracy
mse = mean_squared_error(y_test, prediction)
rmse = np.sqrt(mse)

r2 = r2_score(y_test, prediction)
mae = mean_absolute_error(y_test, prediction)
print(f'Model Performance:')
print(f'R2 score: {r2}')
print(f'Root Mean Square Error: {rmse}')
print(f'Mean Absolute Error: {mae}')

joblib.dump(pipeline, "vehicle_emission_pipeline.joblib")