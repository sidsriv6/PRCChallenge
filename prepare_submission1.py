import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
import joblib
import torch

data = pd.read_csv("challenge_file_nov_dec.csv")
print("Challenge Shape: ", data.shape)

data = data.drop(['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'flight_id', 'callsign', 'name_adep', 'name_ades', 'route', 'adep',
       'name_adep', 'country_code_adep', 'ades', 'name_ades',
       'country_code_ades', 'wtc', 'actual_offblock_time', 'arrival_time', 'local_offblock_time', 'local_arrival_time', 'date', 'aircraft_type', 'airline'], axis=1)
print(data.columns)
#data = data.drop(['cruise_sfc', 'cruise_thrust'])
#data = data.drop(['min_temperature_difference', 'max_temperature_difference'], axis=1)

# Fill missing values in numerical columns with the mean of each column
for col in ['avg_takeoff_groundspeed','avg_takeoff_climb_rate','fuel_rate_climb', 'avg_descent_groundspeed' ,'fuel_rate_descent' ,'fuel_rate_cruise', 'mean_headwind_component', 'max_temperature_difference']:
    data[col].fillna(data[col].mean(), inplace=True)
for col in ['avg_descent_climb_rate','min_headwind_component','mean_temperature_difference','max_headwind_component', 'max_alt_climb', 'min_temperature_difference']:
    data[col].fillna(data[col].median(), inplace=True)
#for col in ['avg_takeoff_groundspeed','avg_takeoff_climb_rate','fuel_rate_climb', 'avg_descent_groundspeed' ,'fuel_rate_descent' ,'fuel_rate_cruise', 'mean_headwind_component']:
#    data[col].fillna(data[col].mean(), inplace=True)
#for col in ['avg_descent_climb_rate','min_headwind_component','mean_temperature_difference','max_headwind_component', 'max_alt_climb']:
#    data[col].fillna(data[col].median(), inplace=True)

print("Shape before NA in challenge: ", data.shape)

data = data.dropna()
print("Shape after replacing with mean in challenge: ", data.shape)

X = data.drop('tow', axis=1)  # Features
y = data['tow']               # Target (e.g., takeoff weight)

categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
numerical_columns = X.select_dtypes(exclude=['object']).columns.tolist()

for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

scaler = StandardScaler()
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [500, 1000, 1500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7, 10],
    'min_child_weight': [1, 5, 10],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

xgb_model = xgb.XGBRegressor(
    #'subsample': 0.6, 'n_estimators': 500, 'min_child_weight': 10, 'max_depth': 10, 'learning_rate': 0.05, 'colsample_bytree': 1.0
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='gpu_hist' if torch.cuda.is_available() else 'auto',
    eval_metric="rmse",  # Specify eval_metric during model initialization
    early_stopping_rounds = 10
    #########################
    #subsample=0.6,
    #n_estimators=500,
    #min_child_weight=10,
    #max_depth=10,
    #learning_rate=0.05,
    #colsample_bytree=1.0,
    #tree_method='gpu_hist' if torch.cuda.is_available() else 'auto',
    #eval_metric="rmse",  # Specify eval_metric during model initialization
    #early_stopping_rounds = 10
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=True
)

y_pred = xgb_model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"RMSE on Test Set: {rmse:.4f}")

# Save the model
joblib.dump(xgb_model, "xgb_model_jan.pkl")

# Load the model
loaded_model = joblib.load("xgb_model_jan.pkl")

# Load the new dataset without the 'tow' column
new_data = pd.read_csv("submission_file_nov_dec.csv")  # Replace with your new dataset path
print("Submission Shape: ", new_data.shape)

# Fill missing values in numerical columns with the mean of each column
for colu in ['avg_takeoff_groundspeed','avg_takeoff_climb_rate','fuel_rate_climb', 'avg_descent_groundspeed' ,'fuel_rate_descent' ,'fuel_rate_cruise', 'mean_headwind_component', 'max_temperature_difference']:
    new_data[colu].fillna(new_data[colu].mean(), inplace=True)
for colu in ['avg_descent_climb_rate','min_headwind_component','mean_temperature_difference','max_headwind_component', 'max_alt_climb', 'min_temperature_difference']:
    new_data[colu].fillna(new_data[colu].median(), inplace=True)
#for colu in ['avg_takeoff_groundspeed','avg_takeoff_climb_rate','fuel_rate_climb', 'avg_descent_groundspeed' ,'fuel_rate_descent' ,'fuel_rate_cruise', 'mean_headwind_component']:
#    new_data[colu].fillna(new_data[colu].mean(), inplace=True)
#for colu in ['avg_descent_climb_rate','min_headwind_component','mean_temperature_difference','max_headwind_component', 'max_alt_climb']:
#    new_data[colu].fillna(new_data[colu].median(), inplace=True)

print("Shape before NA in submission: ", new_data.shape)
new_data = new_data.drop(columns=['tow'], axis=1).fillna(0)

new_data = new_data.drop(['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'flight_id', 'callsign', 'name_adep', 'name_ades', 'route', 'adep',
       'name_adep', 'country_code_adep', 'ades', 'name_ades',
       'country_code_ades', 'wtc', 'actual_offblock_time', 'arrival_time', 'local_offblock_time', 'local_arrival_time', 'date', 'aircraft_type', 'airline'], axis=1)
#new_data = new_data.drop(['min_temperature_difference', 'max_temperature_difference'], axis=1)

# Preprocess the new data in the same way as training data
for col in categorical_columns:
    new_data[col] = le.fit_transform(new_data[col].astype(str))  # Use the same LabelEncoder

print("Shape after replacing with mean in submission: ", new_data.shape)
new_data[numerical_columns] = scaler.transform(new_data[numerical_columns])  # Use the same scaler

# Predict 'tow' for the new dataset
new_data['tow'] = loaded_model.predict(new_data)

# Save the predictions back to a CSV file
new_data.to_csv("predicted_data_nov_dec.csv", index=False)  # Adjust the file name as needed

print(new_data['tow'])

print("Predicted 'tow' values saved to 'predicted_data_nov_dec.csv'")
