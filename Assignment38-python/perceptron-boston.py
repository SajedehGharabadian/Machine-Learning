import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

#prepare dataset
data_houses = load_boston()
data = pd.DataFrame(data_houses.data, columns=data_houses.feature_names)
data['MEDV'] = data_houses.target
data = data.loc[data['AGE']>30]
# print(data)
X_train = np.array(data[['RM', 'AGE']])
Y_train = np.array(data[['MEDV']])
Y_train = Y_train.reshape(-1,1)
n=X_train.shape[0]


#hyper parameters
learing_rate = 0.000001
epochs = 1

#init weight
w = np.random.rand(X_train.shape[1],1)

#plot
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121,projection='3d')
ax1 = fig.add_subplot(122)
Errors = []

x_range = np.arange(X_train[:,0].min(),X_train[:,0].max())
y_range = np.arange(X_train[:,1].min(),X_train[:,1].max())
#Train
for epoch in range(epochs):
    for i in range(n):
        y_pred = np.matmul(X_train[i,:],w)
        e = y_pred - Y_train[i]  #khata yek dade
        
        #update weight
        x = X_train[i,:].reshape(-1, 1)
        w += e * learing_rate * x
        
        #visualization
        Y_pred = np.matmul(X_train,w) 

        x,y = np.meshgrid(x_range,y_range)
        z = w[0]*x + w[1]*y
        ax.clear()
        ax.scatter3D(X_train[:,0],X_train[:,1],Y_train)
        ax.set_xlabel("RM")
        ax.set_ylabel("AGE")
        ax.set_zlabel("MEDV")
        ax.plot_surface(x,y,z,alpha=0.5)


        Error = np.mean(Y_train - Y_pred)  #khata kol dadeha
        Errors.append(Error)
        ax1.clear()
        ax1.plot(Errors)

        plt.pause(0.1)


plt.show()