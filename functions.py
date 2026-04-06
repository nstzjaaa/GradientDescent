import numpy as np

def predict(X, theta):
    h=np.dot(X, theta)
    return h

def cost(X, Y, theta):
    m=X.shape[0]
    h=predict(X, theta)
    J=(1/(2*m))*np.sum((h-Y)**2)
    return J

def gradient_descent(X, Y, theta, alpha, num_iters):
    m=X.shape[0]
    J_history=[]
    for i in range(num_iters):
        h=predict(X, theta)
        theta=theta-alpha* (1/m)*np.dot(X.T,(h-Y))
        J_history.append(cost(X, Y, theta))
    return theta, J_history

def predict(X_test, theta):
    y_pred = np.dot(X_test, theta)
    return y_pred


