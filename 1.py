
import numpy as np
import pandas as pd
import json

#wczytanie danych
data=pd.read_csv('data/insurance.csv')

with open('parameters.json') as f:
    parameters=json.load(f)

alpha=parameters['alpha']
num_iters=parameters['num_iters']

data['sex']=data['sex'].map({'male':1,'female':0})
data['smoker']=data['smoker'].map({'yes':1,'no':0})

X=data[['age', 'sex', 'bmi', 'smoker', 'children']].copy()
Y=data[['charges']].copy()

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X, Y, test_size=0.2, random_state=42)

mean_values=X_train.mean()
std_values=X_train.std()

X_train_norm=(X_train-mean_values)/std_values
X_test_norm=(X_test-mean_values)/std_values

X_train=X_train_norm.values
X_test=X_test_norm.values
Y_train=Y_train.values.reshape(-1,1)
Y_test=Y_test.values.reshape(-1,1)
m=X_train.shape[0]
X_train=np.c_[np.ones(m), X_train]
X_test=np.c_[np.ones(X_test.shape[0]), X_test]

theta=np.zeros((X_train.shape[1],1))

from functions import gradient_descent
theta, cost= gradient_descent(X_train, Y_train, theta, alpha, num_iters)

y_pred=np.dot(X_test, theta)

import matplotlib.pyplot as plt

max_val = max(Y_test.max(), y_pred.max())

plt.scatter(Y_test, y_pred)
plt.plot([0, max_val], [0, max_val], color="black")
plt.xlabel("Wartości rzeczywiste")
plt.ylabel("Wartości przewidywane")
plt.title("Rzeczywiste vs Przewidywane")
plt.show()

ss_res = np.sum((Y_test - y_pred) ** 2)
ss_tot = np.sum((Y_test - np.mean(Y_test)) ** 2)

r2 = 1 - ss_res / ss_tot

print(theta)
print("R²:", r2)

