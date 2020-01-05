#!/usr/bin/env python
# coding: utf-8

import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np 
import os
import pandas as pd
import pickle
import random
import seaborn as sns
import time

from skimage.io import imread, imshow
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix as cm
from sklearn.preprocessing import OneHotEncoder


# ### Load Dataset

# In[94]:


data_path = os.path.join(os.getcwd(), 'data')

def convert_to_dataframe(feature_dict):
    df = pd.DataFrame.from_dict(feature_dict,orient='index')
    df.reset_index(level=0, inplace=True)
    df = df.rename(columns={'index': 'Id'})
    return df

def join_features_and_labels(df1, df2):
    return pd.merge(df1, df2, on = 'Id')
    
# Load training, validation and test features
train_feature_file = os.path.join(data_path, 'Train_Features.pkl')
train_features = pickle.load(open(train_feature_file,'rb'), encoding = "latin1")
train_features = convert_to_dataframe(train_features)

val_feature_file = os.path.join(data_path, 'Val_Features.pkl')
val_features = pickle.load(open(val_feature_file,'rb'), encoding = "latin1")
val_features = convert_to_dataframe(val_features)

test_feature_file = os.path.join(data_path, 'Test_Features.pkl')
test_features = pickle.load(open(test_feature_file,'rb'), encoding = "latin1")
test_features = convert_to_dataframe(test_features)

# Load training and validation labels
train_label_file = os.path.join(data_path, 'Train_Labels.csv')
train_labels = pd.read_csv(train_label_file)

val_label_file = os.path.join(data_path, 'Val_Labels.csv')
val_labels = pd.read_csv(val_label_file)

# Create X_train, X_val, Y_train, Y_test
train_features_labels = join_features_and_labels(train_features, train_labels)
val_features_labels = join_features_and_labels(val_features, val_labels)

X_train = train_features_labels.iloc[:, 1:-1]

Y_train = train_features_labels.iloc[:, -1]
Y_train = pd.DataFrame({'Category': Y_train}).astype(int)

X_val = val_features_labels.iloc[:, 1:-1]

Y_val = val_features_labels.iloc[:, -1]
Y_val = pd.DataFrame({'Category': Y_val}).astype(int)

X_test = test_features.iloc[:, 1:]

# Normalize the features
scaler = preprocessing.Normalizer()
X_all = X_train.append(X_val).append(X_test)
scaler.fit(X_all)

X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)

X_train = pd.DataFrame(X_train)
X_val = pd.DataFrame(X_val)


# In[125]:


class LogisticRegressionSGD:
    
    def __init__(self, n0 = 0.0003, n1 = 1, calc_accuracy = False, 
                 delta = 0.0001, max_epochs = 1000, m = 16, l = 0, verbose = True):
        self.l = l
        self.max_epochs = max_epochs
        self.m = m
        self.n0 = n0
        self.n1 = n1
        self.delta = delta
        self.calc_accuracy = calc_accuracy
        self.X_val = X_val
        self.Y_val = Y_val
        self.epoch = 0
        self.verbose = verbose
        self.epsilon = 1e-10
        
    def __generate_random_permutation(self, i):
        perm = np.arange(0, i)
        np.random.shuffle(perm)
        return perm
    
    def calculate_loss(self, X, y_onehot, prob, theta):
        loss = (-1/self.m) * np.sum(np.multiply(y_onehot.T, np.log(prob + self.epsilon)), axis=None) + (self.l/2)*np.sum(np.square(theta))
        return loss
    
    def calculate_probabilities(self, X, theta):
        theta_x = theta.dot(X.T) #  (k x d+1) X (d+1 x m) = k x m
        theta_x -= np.max(theta_x)
        theta_x = np.exp(theta_x)
        col_sum = np.sum(theta_x, axis = 0)
        theta_x = theta_x / col_sum
        return theta_x
    
    def sgd_step(self, X, y_onehot, epoch):
        prob = self.calculate_probabilities(X, self.theta) # k x m
        grad = (y_onehot.T - prob).dot(X) + (self.l * self.theta) # (k x m) x (m x d+1)
        self.theta = self.theta + (self.n * (grad/self.m))
        
    def fit(self, X, y, X_val = None, Y_val = None):
        X['bias'] = 1 # n x d+1
        self.classes = y.nunique()['Category'] # k
#         self.theta = np.zeros((self.classes, X.shape[1])) # k x d+1
        self.theta = np.random.randn(self.classes, X.shape[1]) * 0.001
        train_loss = []
        val_loss = []
        train_accuracies = []
        val_accuracies = []
        # Create one hot encoding
        enc = OneHotEncoder(handle_unknown='ignore')
        Y_onehot_train = enc.fit_transform(y).todense() # m x k
        
        if X_val is not None:
            Y_onehot_val = enc.fit_transform(Y_val).todense() # m x k
            X_val['bias'] = 1
            
        loss_new_train = None
        loss_new_val = None
        
        for epoch in range(self.max_epochs + 1):
            loss_old_train = loss_new_train
            self.n = self.n0 / (self.n1 + epoch)
            permutation = self.__generate_random_permutation(X.shape[0])
            # Create batches
            for i in range(0, X.shape[0], self.m):
                X_minibatch = X.iloc[permutation[i : i + self.m], :] # m x d+1
                Y_minibatch = Y_onehot_train[permutation[i : i + self.m], :] # m x 1
                self.sgd_step(X_minibatch, Y_minibatch, epoch)
         
            loss_new_train = self.calculate_loss(X, Y_onehot_train, self.calculate_probabilities(X, self.theta), self.theta)
            train_loss.append(loss_new_train)

            if X_val is not None:
                loss_new_val = self.calculate_loss(X_val, Y_onehot_val, self.calculate_probabilities(X_val, self.theta), self.theta)
                val_loss.append(loss_new_val)
            
            if epoch % 100 == 0 and self.verbose is True:
                print('After {} epochs, loss_new_train = {}, loss_old_train={}'.format(epoch + 1, loss_new_train, loss_old_train))
            
            if self.calc_accuracy is True:
                y_pred_train = self.predict(X)
                train_accuracies.append(accuracy_score(y, y_pred_train))
                if X_val is not None:
                    y_pred_val = self.predict(X_val)
                    val_accuracies.append(accuracy_score(Y_val, y_pred_val))
                
        self.epoch = epoch + 1
        if self.verbose is True:
            print('Number of epochs: {}'.format(epoch))
            print('Loss after {} epochs = {}'.format(epoch + 1, loss_new_train))
        
        return train_loss, train_accuracies, val_loss, val_accuracies
            
    def predict(self, X):
        X['bias'] = 1 # n x d+1
        probs = self.calculate_probabilities(X, self.theta) # (k x d+1) x (d+1 x n)
        y_pred = probs.argmax(axis=0)
        return np.array(y_pred[0] + 1).T



lr = LogisticRegressionSGD(10, 5, True)
train_losses, train_accur, val_losses, val_accur = lr.fit(X_train, Y_train, X_val, Y_val)


cm_train = cm(Y_train, lr.predict(X_train))

ax = plt.axes([0, 0, 1, 1])
sns.heatmap(cm_train, annot=True, ax = ax, cmap='Reds', fmt='g'); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['1', '2', '3', '4']); ax.yaxis.set_ticklabels(['1', '2', '3', '4']);


# In[137]:


cm_val = cm(Y_val, lr.predict(X_val))
ax = plt.axes([0, 0, 1, 1])
sns.heatmap(cm_val, annot=True, ax = ax, cmap='Reds', fmt='g');

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['1', '2', '3', '4']); ax.yaxis.set_ticklabels(['1', '2', '3', '4']);




