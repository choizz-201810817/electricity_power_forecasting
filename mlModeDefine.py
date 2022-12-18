import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error, r2_score

def rmse(y_test, pred):
    return np.sqrt(mean_squared_error(y_test, pred))


def trainTestAlgo(X_train, X_test, y_train, y_test, algo):
    algo.fit(X_train, y_train)

    trainPred = algo.predict(X_train)
    trainRmse = rmse(y_train, trainPred)
    trainR2 = r2_score(y_train, trainPred)
    
    testPred = algo.predict(X_test)
    testRmse = rmse(y_test, testPred)
    testR2 = r2_score(y_test, testPred)

    print(f"{algo.__class__.__name__}'s train rmse : {trainRmse}")
    print(f"{algo.__class__.__name__}'s test rmse : {testRmse}")
    print(f"{algo.__class__.__name__}'s train r2 score : {trainR2}")
    print(f"{algo.__class__.__name__}'s test r2 score : {testR2}\n\n")
