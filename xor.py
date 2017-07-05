import numpy as np
import pandas as pd
#initializing the inputs which is a truth table for xor gate and 'y' is the output of truth table
#row of the 'x' means the number of examples we have in a neural network
#no of columns mean,the number of features we have.
#'x' is a (4,2) matrix,means 4 examples and two features.
x=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[1],[1],[0]])
#'seed' so that random weights donot change everytime the program runs
np.random.seed(0)

# Optional, but a good idea to have +ve and -ve weights
theta1=np.random.rand(2,8)-0.5
#8 neurons in our hidden layers or we can also call them features. 
theta2=np.random.rand(8,1)-0.5

# Necessary - the bias terms should have same number of dimensions
# as the layer.

b1=np.zeros(8)
b2=np.zeros(1)

alpha=0.01
#'lamda'regularization term to prevent overfitting,not neccessary for example though.
lamda=0.001

# More iterations than you might think! This is because we have
# so little training data, we need to repeat it a lot.
for i in range(1,40000):
    z1=x.dot(theta1)+b1
    h1=1/(1+np.exp(-z1))
    z2=h1.dot(theta2)+b2
    h2=1/(1+np.exp(-z2))
    #This dz term assumes binary cross-entropy loss
    dz2 = h2-y 
   
    # are the derivative of the sigmoid transfer function. 
    # It converges slower though:
    # dz2 = (h2-y) * h2 * (1-h2)

   =
    dw2 = np.dot(h1.T, dz2)
    db2 = np.sum(dz2, axis=0)

    
    dz1 = np.dot(dz2, theta2.T) * h1 * (1-h1)
    dw1 = np.dot(x.T, dz1)
    db1 = np.sum(dz1, axis=0)

    # The L2 regularisation terms ADD to the gradients of the weights
    dw2 += lamda * theta2
    dw1 += lamda * theta1

    theta1 += -alpha * dw1
    theta2 += -alpha * dw2

    b1 += -alpha * db1
    b2 += -alpha * db2

input1=np.array([[0,0],[1,1],[0,1],[1,0]])
z1=np.dot(input1,theta1)+b1
h1=1/(1+np.exp(-z1))
z2=np.dot(h1,theta2)+b2
h2=1/(1+np.exp(-z2))

print(h2)
	
