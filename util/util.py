"""
This is th util.py file for Logistic_Regression project on Fraud-Detection
"""

__authur: str = "Pouya 'Adrian' Firouzmakan"

__all__ = ['generate_dataset_report', 'pre_processing', 'final_report']

import numpy as np
import pandas as pd
from io import StringIO
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score, classification_report

from Cleaning import Cleaning


def generate_dataset_report(df, config):
    with open(config['data']['data_report_path'], 'w') as file:
        buffer = StringIO()
        df.info(buf=buffer)
        buffer.seek(0)
        file.write("DataFrame Info:\n")
        file.write('\n'.join(buffer))
        file.write("\n\n")
        file.write("=" * 96)

        file.write("\n\n")
        file.write("DataFrame Description (Transposed):\n")
        file.write(df.describe().transpose().to_string())
        file.write("\n\n")
        file.write("=" * 96)

        file.write("\n\n")
        file.write(f"Value Counts for {config['data']['label']}:\n")
        file.write(df[config['data']['label']].value_counts().to_string())
        file.write("\n")



def pre_processing(df):
    """
    :param df:
    :return:
    """
    x = df.drop('is_fraud', axis=1)
    y = df['is_fraud']

    cleaner = Cleaning(drop_columns)
    x = cleaner.fit_transform(x, y)

    up_sampling = SMOTE(random_state=42)
    x, y = up_sampling.fit_resample(x, y) 

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)

    return xtrain, xtest, ytrain, ytest


def final_report(best_estimator, best_params,
                 xtest, ytest, config):
    """
    :param best_estimator:
    :param best_params:
    :param xtest:
    :param ytest:
    :return:
    """
    prediction = best_estimator.predict(xtest)
    lines = list()
    lines.append(f'best_params: {best_params}')
    lines.append(f'best_estimator: {best_estimator}')
    lines.append('='*96)

    # Evaluating our trained model
    lines.append("Confusion Matrix:")
    lines.append(str(confusion_matrix(ytest, prediction)))
    lines.append(str(classification_report(ytest, prediction)))
    lines.append(f"F1 Score: {f1_score(ytest, prediction)}")
    lines.append(f"Accuracy: {accuracy_score(ytest, prediction)}")
    lines.append(f"ROC AUC Score: {roc_auc_score(ytest, prediction)}")

    with open (config['output_path'], 'w') as report:
        for l in lines:
            report.write(l + "\n\n")
