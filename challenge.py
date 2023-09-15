import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# set seed for reproducibility
np.random.seed(0)
        
def outcome(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f'MAE: {mae}')
    print(f'RMSE: {rmse}')
    print(f'R-squared: {r2}')
        
df = pd.read_csv("Dataset_Valmet_sensor_data.csv")
df = df.dropna(subset=['Roughness','Water Flow','Dry Weight'])
df = df.drop(['Filler Amount','Time'], axis=1)

threshold = 3

# calculating Z-scores for all columns of the dataframe
z_scores = np.abs((df - df.mean()) / df.std())

# Detect outliers with Z-score
outliers = df[(z_scores > threshold).any(axis=1)]

# Clean outliers with z-score
df_clean = df[(z_scores <= threshold).all(axis=1)]

#Detect outliers with IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = ((df_clean < lower_bound) | (df_clean > upper_bound))

# Clean outliers with IQR
df_no_outliers = df_clean[~outliers.any(axis=1)]

#Split into training data
X = df_no_outliers.drop('Roughness', axis=1)  # Features (sensordata)
y = df_no_outliers['Roughness']  # Zielvariable (Roughness)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Random Forest model
def randomForest(X_train, y_train, X_test, y_test):
    rf_model = RandomForestRegressor(n_estimators=250, max_depth=30)

    # Training of the model
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    print("Random Forest:")
    outcome(y_test, y_pred)

randomForest(X_train, y_train, X_test, y_test)