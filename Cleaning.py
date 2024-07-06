import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class Cleaning(BaseEstimator, TransformerMixin):

    def __init__(self, drop_columns=None):
        self.drop_columns = drop_columns

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        data = x

        if self.drop_columns:
            data.drop(self.drop_columns, axis=1, inplace=True)

        data['age_year'] = pd.Timestamp.now().year - pd.to_datetime(data['dob']).dt.year
        data.drop('dob', axis=1, inplace=True)

        label_encoder = LabelEncoder()
        for col in ['merchant', 'city', 'state', 'gender', 'job', 'category']:
            data[col] = label_encoder.fit_transform(data[col])

        return data
