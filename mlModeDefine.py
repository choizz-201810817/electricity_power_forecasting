import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import ShuffleSplit, learning_curve

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

def learningCurveDraw(algo, X, y, size, epochs):
    trainSizes, trainScores, testScores = learning_curve(algo, X, y, cv=3, n_jobs=1, train_sizes = np.linspace(.1, size, epochs))
    trainScoresMean = np.mean(trainScores, axis=1)
    testScoresMean = np.mean(testScores, axis=1)
    plt.plot(trainSizes, trainScoresMean, 'o-', color='blue', label='Training score')
    plt.plot(trainSizes, testScoresMean, 'o-', color='red', label='Cross validation score')
    plt.title(f"{algo.__class__.__name__}'s learning curve")
    plt.legend(loc='best')