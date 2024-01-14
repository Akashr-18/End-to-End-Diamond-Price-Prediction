import numpy as np
from DiamondRegressor import logger
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from DiamondRegressor.entity.config_entity import ModelParametersConfig
from DiamondRegressor.constants import *
from DiamondRegressor.utils.common import read_yaml
import mlflow
from urllib.parse import urlparse

class ModelFinder:
    def __init__(self, 
                 params_filepath = PARAMS_FILE_PATH):
        
        self.params = read_yaml(params_filepath)

    def xgboost_model(self, x_train, x_test, y_train, y_test):
        config = read_yaml(PARAMS_FILE_PATH)
        xgb_model = XGBRegressor()
        params={
            "n_estimators"     : self.params['models']['XgBoost']['params']['n_estimators'],  #No of gradient boosted trees.Equivalent to no of boosting rounds.
            "learning_rate"    : self.params['models']['XgBoost']['params']['learning_rate'], #Boosting learning rate
            "max_depth"        : self.params['models']['XgBoost']['params']['max_depth'],  #Maximum tree depth for base learners.
            "gamma"            : self.params['models']['XgBoost']['params']['gamma'], #(min_split_loss) Minimum loss reduction required to make a further partition on a leaf node of the tree.
            "min_child_weight" : self.params['models']['XgBoost']['params']['min_child_weight'],  #Minimum sum of instance weight(hessian) needed in a child.
            # "colsample_bytree" : config['models']['XgBoost']['params']['colsample_bytree'],  #Subsample ratio of columns when constructing each tree.
            # 'subsample'        : config['models']['XgBoost']['params']['subsample'], #Setting it to a value less than 1.0 may help prevent overfitting.
            # 'reg_alpha'        : config['models']['XgBoost']['params']['reg_alpha'],
            # 'reg_lambda'       : config['models']['XgBoost']['params']['reg_lambda'], #L1 (Lasso) and L2 (Ridge) regularization terms. These can help control overfitting by penalizing large coefficients.
            }
        random_search_xgb = RandomizedSearchCV(xgb_model, param_distributions=params, n_iter=5, n_jobs=-1, cv=5)
        random_search_xgb.fit(x_train,y_train)
        logger.info(f"Best params for XGBoost: {random_search_xgb.best_params_}")
        updated_xgb_model = random_search_xgb.best_estimator_
        updated_xgb_model.fit(x_train,y_train)
        y_pred_xgb = updated_xgb_model.predict(x_test)
        score_r2_xgb = r2_score(y_test, y_pred_xgb)
        mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
        mse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
        return updated_xgb_model, score_r2_xgb, mae_xgb, mse_xgb
    
    def randomforest_model(self, x_train, x_test, y_train, y_test):
        rf_model = RandomForestRegressor()
        params = {
            'n_estimators'      : self.params['models']['RandomForest']['params']['n_estimators'],
            'criterion'         : self.params['models']['RandomForest']['params']['criterion'],
            'max_depth'         : self.params['models']['RandomForest']['params']['max_depth'],
            'min_samples_split' : self.params['models']['RandomForest']['params']['min_samples_split'],
            # 'min_samples_leaf'  : self.params['models']['RandomForest']['params']['min_samples_leaf'],
            # 'bootstrap'         : self.params['models']['RandomForest']['params']['bootstrap'],
        }
        random_search_rf = RandomizedSearchCV(rf_model, param_distributions=params, n_iter=5, n_jobs=-1, cv=5)
        random_search_rf.fit(x_train, y_train)
        logger.info(f"Best params for RandomForest: {random_search_rf.best_params_}")
        updated_rf_model = random_search_rf.best_estimator_
        updated_rf_model.fit(x_train,y_train)
        y_pred_rf = updated_rf_model.predict(x_test)
        score_r2_rf = r2_score(y_test, y_pred_rf)
        mae_rf = mean_absolute_error(y_test, y_pred_rf)
        mse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
        return updated_rf_model, score_r2_rf, mae_rf, mse_rf

    def decisiontree_model(self, x_train, x_test, y_train, y_test):
        model = DecisionTreeRegressor()
        params = {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=5, n_jobs=-1, cv=5)
        random_search.fit(x_train, y_train)
        print("Best params for Decision Tree: ", random_search.best_params_)
        updated_model = random_search.best_estimator_
        updated_model.fit(x_train,y_train)
        y_pred = updated_model.predict(x_test)
        score_r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return updated_model, score_r2, mae, rmse
    
    def svr_model(self, x_train, x_test, y_train, y_test):
        model = SVR()
        params = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto'],
        }
        random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=5, n_jobs=-1, cv=5)
        random_search.fit(x_train, y_train)
        print("Best params for SVR: ", random_search.best_params_)
        updated_model = random_search.best_estimator_
        updated_model.fit(x_train,y_train)
        y_pred = updated_model.predict(x_test)
        score_r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return updated_model, score_r2, mae, rmse
    
    def linearregression_model(self, x_train, x_test, y_train, y_test):
        updated_model = LinearRegression()
        updated_model.fit(x_train,y_train)
        y_pred = updated_model.predict(x_test)
        score_r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return updated_model, score_r2, mae, rmse
    
    def train_and_log_model(model, train_x, train_y, test_x, test_y):
        with mlflow.start_run():
            logger.info(f"MLflow train model started")
            model.fit(train_x, train_y)
            pred_y = model.predict(test_x)

            r2 = r2_score(test_y, pred_y)
            mae = mean_absolute_error(test_y, pred_y)
            rmse = np.sqrt(mean_squared_error(test_y, pred_y))

            mlflow.log_metric("r2", r2)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)

            # For remote server only (Dagshub)
            remote_server_uri = "https://dagshub.com/Akashr-18/mlflow-demo.mlflow"
            mlflow.set_tracking_uri(remote_server_uri)

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            model_name_convention = type(model).__name__
            if tracking_url_type_store != "file":
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(
                    model,
                    "model", 
                    registered_model_name=model_name_convention
                )
            else:
                mlflow.sklearn.log_model(model, "model")

            # Save the model to the prediction_service/model directory
            # model_path = "prediction_service/model"
            # mlflow.pyfunc.save_model(model, model_path)

            # mlflow.sklearn.log_model(model, "model")

            model_name = type(model).__name__.lower()
            model_path = f"service/model_{model_name}"
            mlflow.pyfunc.save_model(model, model_path)

    @staticmethod
    def find_best_model_for_cluster(x_train, x_test, y_train, y_test):
        lr_model, lr_r2_score, lr_mae, lr_rmse = ModelFinder().linearregression_model(x_train, x_test, y_train, y_test) 
        # svr_model, svr_r2_score, svr_mae, svr_rmse = self.svr_model(x_train, x_test, y_train, y_test)
        # dt_model, dt_r2_score, dt_mae, dt_rmse = self.decisiontree_model(x_train, x_test, y_train, y_test)
        xgb_model, xgb_r2_score, xgb_mae, xgb_rmse = ModelFinder().xgboost_model(x_train, x_test, y_train, y_test)
        # rf_model, rf_r2_score, rf_mae, rf_rmse = ModelFinder().randomforest_model(x_train, x_test, y_train, y_test)
        # print('LR: ', lr_r2_score, lr_mae, lr_rmse)
        # print('SVR: ', svr_r2_score, svr_mae, svr_rmse)
        # print('DT: ', dt_r2_score, dt_mae, dt_rmse)
        logger.info(f"XGB model is {xgb_model}")
        logger.info(f"RF model is {lr_model}")
        ModelFinder().train_and_log_model(lr_model,x_train, x_test, y_train, y_test)
        ModelFinder().train_and_log_model(xgb_model,x_train, x_test, y_train, y_test)
        # logger.info(f"Eval Metrics for Random Forest model r2score: {rf_r2_score} mae: {rf_mae}, rmse: {rf_rmse}")
        logger.info(f"Eval Metrics for XgBoost model r2score: {xgb_r2_score} mae: {xgb_mae}, rmse: {xgb_rmse}")

        model_r2_scores = {
            'linear_regression': {'model': lr_model, 'r2_score': lr_r2_score},
            # 'random_forest': {'model': rf_model, 'r2_score': rf_r2_score},
            'xgboost': {'model': xgb_model, 'r2_score': xgb_r2_score},
        }
        best_model_info = max(model_r2_scores.values(), key=lambda x: x['r2_score'])
        best_model = best_model_info['model']
        best_r2_score = best_model_info['r2_score']
        logger.info("Comparing DecisionTree, RandomForest, and XGBoost models")
        logger.info(f"The best model is {type(best_model).__name__} with an R-squared score of {best_r2_score}")
        model_name = type(best_model).__name__
        return best_model, model_name

    