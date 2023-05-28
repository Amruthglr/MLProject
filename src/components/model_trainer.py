import numpy as np
import pandas as pd
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from exception import CustomException
from logger import logging
from utils import save_object, evaluate_models

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()
    
    def initate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test data")

            X_train, y_train, X_test, y_test = (train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1])

            models = {
                "Random_forest": RandomForestRegressor(),
                "Decision_tree": DecisionTreeRegressor(),
                "Gradient_Boost": GradientBoostingRegressor(),
                "Linear_Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoost_Regressor": CatBoostRegressor(verbose=False),
                "Adaboost_Regressor": AdaBoostRegressor()
            }

            params = {
                "Random_forest":{
                    'n_estimators': [8,16,32,64,128,256],
                    'max_depth': [3,5,10],
                    'criterion' : ["squared_error", "absolute_error", "friedman_mse", "poisson"],
                    'max_features' : ['sqrt', 'log2', None]
                },
                "Decision_tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                'Gradient_Boost': {
                    'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'criterion':['squared_error', 'friedman_mse'],
                    'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                'Linear_Regression': {},
                'XGBRegressor': {
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoost_Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "Adaboost_Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                 }
            }

            model_report:dict = evaluate_models(X_train =X_train, y_train= y_train, X_test= X_test, y_test = y_test, models = models, param = params)
            print(type(model_report))
            # To get the best model score
            best_score = max(sorted(list(model_report.values())))

            # To ge the best model name
            model_report = {k: v for k, v in sorted(model_report.items(), key=lambda item: item[1])}
            best_model_name = list(model_report.keys())[-1]
            best_model = models[best_model_name]

            if best_score < 0.6:
                raise CustomException("No good model found")
            logging.info("Best found model on training and test data")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e,sys)
            