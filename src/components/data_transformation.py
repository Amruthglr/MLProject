import numpy as np
import pandas as pd
import os
import sys
from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from exception import CustomException
from logger import logging

from utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transform_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        
        try:
            num_features = ['reading score', 'writing score']
            cat_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            num_feature_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler(with_mean=False))
                ]
            )

            cat_feature_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ("one_hot_endcoder", OneHotEncoder() ),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical features {num_features}")
            logging.info(f"Catergorical features {cat_features}")
            
            preprocessor = ColumnTransformer(
                [('num_feature', num_feature_pipeline, num_features),
                 ("cat_feature", cat_feature_pipeline, cat_features)
                 ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data")
            logging.info("Obtaining preprocessing object")

            pre_processing_obj = self.get_data_transformer_obj()

            target_column_name = 'math score'
            numerical_columns = ['reading score', 'writing score']

            input_feature_train_df = train_df.drop(columns=['math score'], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=['math score'], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessing object on train and test data")

            input_feature_train_arr = pre_processing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = pre_processing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("saving preprocessoed data")

            save_object(

                file_path=self.data_transform_config.preprocessor_obj_file_path,
                obj=pre_processing_obj

            )

            return (train_arr, test_arr, self.data_transform_config.preprocessor_obj_file_path)
        except Exception as e:
            raise CustomException(e, sys)
        

