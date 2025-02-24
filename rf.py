import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv("train_final.csv")

# preprocessing - first we start with imputing NaN values in numerical columns 
numerical_cols = df.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='median')
df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

# now, we can handle categorical columns, using the mode for NaN values
categorical_cols = df.select_dtypes(include=[object]).columns
imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = imputer.fit_transform(df[categorical_cols])

# we encode columns containing categorical variables. They do not need to be OHE, as we are working with random forests. 
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# extract our dependent and independent variables
X = df[df.columns[:-1]]
y = df[df.columns[-1]]

# create the random forest
rf_model = RandomForestRegressor(n_estimators=40, random_state=42)
rf_model.fit(X, y)

with open('random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)
print("Model saved successfully.")