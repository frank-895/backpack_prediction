import pickle
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

with open('random_forest_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

df = pd.read_csv('test.csv')

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
    
expected_columns = list(pd.read_csv("train_final.csv").columns[:-1])
df["Unnamed: 0"] = df.index

x_test = df[df.columns[1:]]
x_test = x_test[expected_columns]

y_preds = loaded_model.predict(x_test)

subm = df[['id']].copy()
subm['Price'] = y_preds

subm.to_csv("submission.csv", index=False)