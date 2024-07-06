"""
This is th util.py file for Logistic_Regression project on Fraud-Detection
"""

__authur: str = "Pouya 'Adrian' Firouzmakan"

__all__ = ['generate_dataset_report', 'pre_processing', 'final_report']

from io import StringIO
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import (confusion_matrix, f1_score, classification_report,
                             accuracy_score, roc_auc_score)
from Cleaning import Cleaning
from config.config import config


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


def pre_processing(df, drop_columns):
    """
    :param df:
    :return:
    """

    x = df.drop('is_fraud', axis=1)
    y = df['is_fraud']

    cleaner = Cleaning(drop_columns)
    x = cleaner.fit_transform(x, y)

    sampler = None
    if config['data']['sampling'] == 'up':
        sampler = SMOTE(random_state=config['seed'])
    elif config['data']['sampling'] == 'under':
        sampler = RandomUnderSampler(random_state=config['seed'])
    x, y = sampler.fit_resample(x, y)

    xtrain, xtest, ytrain, ytest = train_test_split(x, y,
                                                    test_size=config['data']['test_size'],
                                                    random_state=config['seed'])

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
    lines.append((f"sampling type: {config['data']['sampling']}-sampling"))
    lines.append(f'best_params: {best_params}')
    lines.append(f'best_estimator: {best_estimator}')
    lines.append('=' * 96)

    # Evaluating our trained model
    lines.append("Confusion Matrix:")
    lines.append(str(confusion_matrix(ytest, prediction)))
    lines.append(str(classification_report(ytest, prediction)))
    lines.append(f"F1 Score: {f1_score(ytest, prediction)}")
    lines.append(f"Accuracy: {accuracy_score(ytest, prediction)}")
    lines.append(f"ROC AUC Score: {roc_auc_score(ytest, prediction)}")

    with open(config['output_path'], 'w') as report:
        for l in lines:
            report.write(l + "\n\n")
