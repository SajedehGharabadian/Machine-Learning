import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split



class Perceptron:
    def __init__(self,lr=0.05,epoch=2):
        self.learning_rate = lr
        self.epoch = epoch

    def fit(self,X_train,Y_train):
        self.w = np.random.rand(X_train.shape[1],1)
        self.X_train = np.array(X_train)
        self.Y_train = np.array(Y_train)
        Errors = []
        for epoch in range(self.epoch):
            for i in range(self.X_train.shape[0]):
                y_pred = np.matmul(X_train[i,:],self.w)
                e = y_pred - Y_train[i]  #khata yek dade
                x = X_train[i,:].reshape(-1, 1)
                self.w += e * self.learning_rate * x
                
                #visualization
                Y_pred = np.matmul(self.X_train,self.w)

                Error = np.mean(Y_train - Y_pred)  #khata kol dadeha
                Errors.append(Error) 
        return Errors

    def predict(self,X_test):
        y_pred = np.matmul(X_test,self.w)

        return y_pred

    def evaluate(self,X,Y):
        Y_pred = self.predict(X)
        Error = Y - Y_pred

        return np.mean(Error ** 2)


            

    
data_houses = load_boston()
data = pd.DataFrame(data_houses.data, columns=data_houses.feature_names)
data['MEDV'] = data_houses.target
data = data.loc[data['AGE']>30]
# print(data)
X = np.array(data[['RM', 'AGE']])
Y = np.array(data[['MEDV']])
Y = Y.reshape(-1,1)
perceptron = Perceptron(0.000001,1)
X_train, X_test, y_train, y_test = train_test_split(X,Y)
perceptron.fit(X_train,y_train)


MSE_train = perceptron.evaluate(X_train,y_train)

print(MSE_train)

MSE_test = perceptron.evaluate(X_test,y_test)

print(MSE_test)
