import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import matplotlib
import os
import matplotlib.ticker as ticker
import random
import time
import math

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


data_path = os.path.join(os.getcwd(), 'data')

# we will first load the training data
train_data = pd.read_csv(os.path.join(data_path, 'trainData.csv'), header=None, index_col=0)
train_labels = pd.read_csv(os.path.join(data_path, 'trainLabels.csv'), header=None, usecols=[1])

print(train_data.shape)


# load the validation data
val_data = pd.read_csv(os.path.join(data_path, 'valData.csv'), header=None, index_col=0)
val_labels = pd.read_csv(os.path.join(data_path, 'valLabels.csv'), header=None, usecols=[1])

#load the test data
test_data = pd.read_csv(os.path.join(data_path, 'testData_new.csv'), header=None, index_col=0)

print(val_data.shape)
print(val_labels.shape)


X = np.concatenate((train_data, val_data), axis = 0) # n x d
Y = np.concatenate((train_labels, val_labels), axis = 0)

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.5, shuffle=False)

ones = np.ones((X_test.shape[0], 1))
X_test = np.concatenate((X_test, ones), axis = 1)


def runRidge(X, y, lmbda, runLoocv = False):
    loocv_error = 0   
    # add 1 to the end of training data
    ones = np.ones((X.shape[0], 1))
    X = np.concatenate((X, ones), axis = 1) # n x (d+1)
    # create an identity matrix
    I = np.identity(X.shape[1])
    I[I.shape[0]-1][I.shape[1]-1] = 0
    
    # use the closed form solution to compute w
    d = X.T.dot(y) # (d+1) x n * n x 1 = (d+1) x 1
    C = (X.T).dot(X) + I*lmbda # (d+1) x n
    w = np.linalg.solve(C, d) # d+1 x 1

    tmp = (w.T).dot(X.T) - y.T # (1 x d+1) x (d+1 x n) = 1 x n
    objective_cost = (w.T).dot(w)*lmbda + tmp.dot(tmp.T)*lmbda # (1 x d+1) x (d+1 x 1) + (1 x n) x (n x 1)
    start_time = time.time()
    cvErrors = []
    if runLoocv is True:
        # starting LOOCV
        for val_idx in range(X.shape[0]):
            x_i = X[val_idx, :]
            x_i = x_i.reshape(1, X.shape[1]).T # x_i = d+1 x 1
            temp = np.linalg.solve(C, x_i) # temp = d+1 x 1
            numerator = (w.T.dot(x_i) - y[val_idx,:])
            denominator = (1 - (x_i.T).dot(temp))
            loocv_error = numerator/denominator # (1 x d+1) x (d+1 x 1) - 1/ 1 - (1x d+1) x (d+1 x 1)
            cvErrors.append(loocv_error[0][0])
            if val_idx%1000 == 0:
                print('LOOCV iterations done={}'.format(val_idx))
    end_time = time.time()
    print(end_time - start_time)
    bias = w[w.shape[0] - 1, 0]
    w = np.delete(w, w.shape[0] - 1, 0)
    return w, bias, objective_cost[0][0], cvErrors


# In[89]:


lmbdas = [0.01, 0.1, 1, 10, 100, 1000]

rmse_train_list = []
rmse_validation_list = []
rmse_loocv_list = []

best_weights = None
best_obj_value = None

for lmbda in lmbdas:
    w_train, b_train, obj_train, cvErrors_train = runRidge(X_train, y_train, lmbda, False)
    rmse_train = math.sqrt(mean_squared_error(y_train, X_train.dot(w_train) + b_train))
    rmse_train_list.append(rmse_train)
    y_val_predicted = X_val.dot(w_train) + b_train
    rmse_validation = math.sqrt(mean_squared_error(y_val, y_val_predicted))
    rmse_validation_list.append(rmse_validation)
    rmse_loocv_list.append(math.sqrt(np.sum(np.square(cvErrors_train))/len(cvErrors_train)))
    print('lambda={}. rmse_train={}, rmse_validation={}, rmse_loocv={}'.format(lmbda, rmse_train, rmse_validation, rmse_loocv_list[-1]))


plt.plot(lmbdas,rmse_train_list,'-bo')
plt.xlabel('lambda')
plt.ylabel('RMSE Train')

plt.plot(lmbdas,rmse_validation_list,'-or')
plt.xlabel('lambda')
plt.ylabel('RMSE Val')

plt.plot(lmbdas,rmse_loocv_list,'-og')
plt.xlabel('lambda')
plt.ylabel('RMSE')

plt.legend(['RMSE Training data', 'RMSE Validation data', 'RMSE LOOCV'])
