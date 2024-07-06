"""
This project is to apply classification on credit-card fraud
"""
__author__ = "Pouya 'Adrian' Firouzmakan"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix, f1_score, classification_report, accuracy_score, auc

from Cleaning import Cleaning
import util.util as util
from config.config import config

if __name__ == '__main__':

    train_data = pd.read_csv('/Users/pouyafirouzmakan/Desktop/fraudTrain.csv')
    test_data = pd.read_csv('/Users/pouyafirouzmakan/Desktop/fraudTest.csv')
    df = pd.concat([train_data, test_data],axis=0).sample(frac=1, random_state=42)

    util.generate_dataset_report(df, config)

    drop_columns = ['first', 'last', 'cc_num', 'Unnamed: 0', 'zip', 'street', 'unix_time',
                    'merch_lat','merch_long', 'trans_num', 'trans_date_trans_time', 'merchant'] 

    plt.figure(dpi=100, figsize=(8,6))
    sns.countplot(data = df, x='is_fraud', hue = 'is_fraud')
    plt.show()

    xtrain, xtest, ytrain, ytest = util.pre_processing(df)

    scaler = StandardScaler()
    model = RandomForestClassifier(n_estimators=100, max_features=3)

    operations = [
        ('scaler', scaler), 
        ('model', model)
                 ]

    pipe = Pipeline(steps=operations)


    params = {}

    grid_model = GridSearchCV(estimator=pipe, param_grid=params,
                              scoring='f1', cv=10, verbose=2)

    grid_model.fit(xtrain_res, ytrain_res)


    final_report(grid_model.best_estimator_, grid_model.best_params_,
                     xtest, ytest, config)