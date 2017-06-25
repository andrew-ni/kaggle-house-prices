import numpy as np 
from datacleaning3 import read_file
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

def linear_regression(Xtrain, Xtest, ytrain, ytest):
  clf = LinearRegression()
  clf.fit(Xtrain, ytrain)

  predictions = clf.predict(Xtest)
  error = predictions - ytest

  return predictions, error, np.mean(error**2), clf.coef_

def linear_regression_feature_selection(Xtrain, Xtest, ytrain, ytest, k):
    sel = SelectKBest(chi2, k=k)
    Xtrain = sel.fit_transform(Xtrain, ytrain)
    Xtest = sel.transform(Xtest)
    return linear_regression(Xtrain, Xtest, ytrain, ytest)

def gradient_descent(Xtrain, Xtest, ytrain, ytest):
  clf = SGDRegressor(alpha=0.0000001)
  clf.fit(Xtrain, ytrain)

  predictions = clf.predict(Xtest)
  error = predictions - ytest

  return predictions, error, np.mean(error**2), clf.coef_

def random_forest(Xtrain, Xtest, ytrain, ytest):
  clf = RandomForestRegressor()
  clf.fit(Xtrain, ytrain)

  predictions = clf.predict(Xtest)
  error = predictions - ytest

  return predictions, error, np.mean(error**2)

def main():
  features, target = read_file('train.csv')
  Xtrain, Xtest, ytrain, ytest = train_test_split(features, target, test_size=0.2, random_state=42)

  print("Linear Regression (OLS):")
  predictions, error, mse, weights = linear_regression(Xtrain, Xtest, ytrain, ytest)
  for i in range(len(ytest)):
    print("%.2lf - %.2lf" % (predictions[i], ytest[i]))
  print("MSE: ", mse)

  print("Random Forest Regressor:")
  predictions, error, mse = random_forest(Xtrain, Xtest, ytrain, ytest)
  for i in range(len(ytest)):
    print("%.2lf %.2lf" % (predictions[i], ytest[i])) 
  print("MSE: ", mse)

  print("Gradient Descent (SGD):")
  predictions, error, mse, weights = gradient_descent(Xtrain, Xtest, ytrain, ytest)
  for i in range(len(ytest)):
    print("%.2lf %.2lf" % (predictions[i], ytest[i]))
  print("MSE: ", mse)

  print("Linear Regression (OLS) with Feature Selection:")
  feat_nums = list()
  for i in range(1, 301):
    feat_nums.append(i)
  lowest_mse = float('inf')
  best_feat_num = 0
  X = list()
  y = list()
  for feat_num in feat_nums:
    predictions, error, mse, weights = linear_regression_feature_selection(Xtrain, Xtest, ytrain, ytest, feat_num)
    print("MSE:", mse, "Number of features:", feat_num)
    X.append(feat_num)
    y.append(mse)
    if lowest_mse > mse:
      lowest_mse = mse
      best_feat_num = feat_num 
  print("Lowest MSE:", lowest_mse, "Best Number of Features:", best_feat_num)
  plt.plot(X, y)
  plt.xlabel('Number of Features')
  plt.ylabel('MSE')
  plt.title('Linear Regression (OLS) with Feature Selection')
  plt.show()

if __name__ == '__main__':
  main()