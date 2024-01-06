import numpy as np
from DiamondRegressor import logger
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

class ModelFinder:
    def __init__(self):
        pass

    def xgboost_model(self, x_train, x_test, y_train, y_test):
        xgb_model = XGBRegressor()
        params={
            "n_estimators"     : [50, 100, 150, 200],  #No of gradient boosted trees.Equivalent to no of boosting rounds.
            "learning_rate"    : [0.05, 0.10, 0.15, 0.20] , #Boosting learning rate
            "max_depth"        : [ 3, 4, 6, 8],  #Maximum tree depth for base learners.
            "min_child_weight" : [ 1, 2, 3, 4],  #Minimum sum of instance weight(hessian) needed in a child.
            "gamma"            : [ 0.0, 0.1, 0.2 , 0.3], #(min_split_loss) Minimum loss reduction required to make a further partition on a leaf node of the tree.
            "colsample_bytree" : [ 0.5 , 0.7, 0.8, 0.9 ],  #Subsample ratio of columns when constructing each tree.
            'subsample'        : [0.8, 0.9, 1.0], #Setting it to a value less than 1.0 may help prevent overfitting.
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 1.0, 10.0, 50.0], #L1 (Lasso) and L2 (Ridge) regularization terms. These can help control overfitting by penalizing large coefficients.
            }
        random_search_xgb = RandomizedSearchCV(xgb_model, param_distributions=params, n_iter=5, n_jobs=-1, cv=5)
        random_search_xgb.fit(x_train,y_train)
        print("Best params for XGBoost: ", random_search_xgb.best_params_)
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
            'n_estimators': [50, 100, 150, 200],
            'criterion': ["squared_error", "absolute_error", "poisson"],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False],
        }
        random_search_rf = RandomizedSearchCV(rf_model, param_distributions=params, n_iter=5, n_jobs=-1, cv=5)
        random_search_rf.fit(x_train, y_train)
        print("Best params for RandomForest: ", random_search_rf.best_params_)
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
    
    @staticmethod
    def find_best_model_for_cluster(x_train, x_test, y_train, y_test):
        logger.info("XXXXXXXXXXXXX")
        # lr_model, lr_r2_score, lr_mae, lr_rmse = self.linearregression_model(x_train, x_test, y_train, y_test) 
        # svr_model, svr_r2_score, svr_mae, svr_rmse = self.svr_model(x_train, x_test, y_train, y_test)
        # dt_model, dt_r2_score, dt_mae, dt_rmse = self.decisiontree_model(x_train, x_test, y_train, y_test)
        rf_model, rf_r2_score, rf_mae, rf_rmse = ModelFinder().randomforest_model(x_train, x_test, y_train, y_test)
        xgb_model, xgb_r2_score, xgb_mae, xgb_rmse = ModelFinder().xgboost_model(x_train, x_test, y_train, y_test)
        # print('LR: ', lr_r2_score, lr_mae, lr_rmse)
        # print('SVR: ', svr_r2_score, svr_mae, svr_rmse)
        # print('DT: ', dt_r2_score, dt_mae, dt_rmse)
        print('RF: ', rf_r2_score, rf_mae, rf_rmse)
        print('XGB: ', xgb_r2_score, xgb_mae, xgb_rmse)

        model_r2_scores = {
            # 'decision_tree': {'model': dt_model, 'r2_score': dt_r2_score},
            'random_forest': {'model': rf_model, 'r2_score': rf_r2_score},
            'xgboost': {'model': xgb_model, 'r2_score': xgb_r2_score},
        }
        best_model_info = max(model_r2_scores.values(), key=lambda x: x['r2_score'])
        best_model = best_model_info['model']
        best_r2_score = best_model_info['r2_score']
        logger.info("Comparing DecisionTree, RandomForest, and XGBoost models")
        logger.info(f"The best model is {type(best_model).__name__} with an R-squared score of {best_r2_score}")
        model_name = type(best_model).__name__
        return best_model, model_name

    # @staticmethod
    # def find_best_model_for_cluster(self, x_train, x_test, y_train, y_test):
    #     logger.info("XXXXXXXXXXXXX")
    #     # lr_model, lr_r2_score, lr_mae, lr_rmse = self.linearregression_model(x_train, x_test, y_train, y_test) 
    #     # svr_model, svr_r2_score, svr_mae, svr_rmse = self.svr_model(x_train, x_test, y_train, y_test)
    #     # dt_model, dt_r2_score, dt_mae, dt_rmse = self.decisiontree_model(x_train, x_test, y_train, y_test)
    #     rf_model, rf_r2_score, rf_mae, rf_rmse = self.randomforest_model(x_train, x_test, y_train, y_test)
    #     xgb_model, xgb_r2_score, xgb_mae, xgb_rmse = self.xgboost_model(x_train, x_test, y_train, y_test)
    #     # print('LR: ', lr_r2_score, lr_mae, lr_rmse)
    #     # print('SVR: ', svr_r2_score, svr_mae, svr_rmse)
    #     # print('DT: ', dt_r2_score, dt_mae, dt_rmse)
    #     print('RF: ', rf_r2_score, rf_mae, rf_rmse)
    #     print('XGB: ', xgb_r2_score, xgb_mae, xgb_rmse)

    #     model_r2_scores = {
    #         # 'decision_tree': {'model': dt_model, 'r2_score': dt_r2_score},
    #         'random_forest': {'model': rf_model, 'r2_score': rf_r2_score},
    #         'xgboost': {'model': xgb_model, 'r2_score': xgb_r2_score},
    #     }
    #     best_model_info = max(model_r2_scores.values(), key=lambda x: x['r2_score'])
    #     best_model = best_model_info['model']
    #     best_r2_score = best_model_info['r2_score']
    #     logger.info(f"Comparing DecisionTree, RandomForest and XGBoost models")
    #     logger.info(f"The best model is {type(best_model).__name__} with an R-squared score of {best_r2_score}")
    #     model_name = type(best_model.__name__)
    #     return best_model, model_name