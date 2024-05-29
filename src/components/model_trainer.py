import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_training(self,train_arr,test_arr,preprocessor_path):
        try:
            logging.info("splitting training and test input data")
            X_train,y_train,X_test,y_test=(train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1])

            models={
                "RandomForest":RandomForestRegressor(),
                "DecisionTree":DecisionTreeRegressor(),
                "GradientBoosting":GradientBoostingRegressor(),
                "LinearRegression":LinearRegression(),
                "KNeighborRegressor":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                "Catboost Regressor":CatBoostRegressor(verbose=False),
                "AdaboostRegressor":AdaBoostRegressor()
            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_value=max(list(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_value)]
            best_model=models[best_model_name]

            if(best_model_value<0.6):
                raise CustomException("No best model found")
            
            logging.info("best model found")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2score=r2_score(y_test,predicted)
            return r2score

        except Exception as e:
            raise CustomException(e,sys)
        
    

