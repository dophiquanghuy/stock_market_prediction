from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,r2_score
from utils import evaluation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt










def linear_regression(x_train, x_test, y_train, y_test, test_pred):
    model_lnr = LinearRegression()
    model_lnr.fit(x_train, y_train)

    y_pred = model_lnr.predict(x_test)

    evaluation(y_test=y_test, y_pred=y_pred, test_pred=test_pred)

    





def support_vector_regression(x_train, x_test, y_train, y_test, test_pred):
    model_svr = SVR()
    model_svr.fit(x_train, y_train)

    y_pred = model_svr.predict(x_test)

    evaluation(y_test=y_test, y_pred=y_pred, test_pred=test_pred)





def gradient_boosting_regression(x_train, x_test, y_train, y_test, test_pred):
    model_gb = GradientBoostingRegressor()
    model_gb.fit(x_train, y_train)

    y_pred = model_gb.predict(x_test)

    evaluation(y_test=y_test, y_pred=y_pred, test_pred=test_pred)





def random_forest_regression(x_train, x_test, y_train, y_test, test_pred):
    model_rf = RandomForestRegressor()
    model_rf.fit(x_train, y_train)

    y_pred = model_rf.predict(x_test)

    evaluation(y_test=y_test, y_pred=y_pred, test_pred=test_pred)