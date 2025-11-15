import pickle

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline

def load_data():
    # Set the path to the file you'd like to load
    file_path = "HR_Data_MNC_Data Science Lovers.csv"

    # Load the latest version
    df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "rohitgrewal/hr-data-mnc",
    file_path,
    )
    df = df.sample(frac=0.3, random_state=42)
    df = df.reset_index(drop=True)
    df = df.drop(['Unnamed: 0', 'Employee_ID', 'Full_Name'], axis=1)
    df.columns = df.columns.str.lower().str.replace(' ','_')

    strings = list(df.dtypes[df.dtypes == 'object'].index)
    for col in strings:
        df[col] = df[col].str.lower().str.replace(' ','_')

    df['location'] = df['location'].str.split(',_').str[-1]
    df['salary_vnd'] = round(df['salary_inr'] * 296.77, 0)

    del df['salary_inr']
    del df['hire_date']

    return df

def train_model(df):
    y = np.log1p(df['salary_vnd'])
    del df['salary_vnd']
    train_dicts = df.to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    dv.fit(train_dicts)
    X_train = dv.transform(train_dicts)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)

    xgb_params = {
            'eta': 0.3, 
            'max_depth': 10,
            'min_child_weight': 1,

            'objective': 'reg:squarederror',
            'nthread': 8,
            'eval_metric': 'rmse',

            'seed': 42,
            'verbosity': 1,
        }

    model = xgb.train(xgb_params, dtrain, num_boost_round=81, verbose_eval=5)

    return model


