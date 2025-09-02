
import numpy as np
from math import exp
from sympy import Eq, solve,symbols
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1.0/(1.0+np.exp(-x))

def train(x,y,learning_rate=0.1, epochs=100):
  w=-9
  b= -8
  initial_weights= (w,b)
  losses=[]

  for i in range(epochs):
    z= x*w+b
    activation= sigmoid(z)
    loss= -np.mean(y * np.log(activation) + (1 - y) * np.log(1 - activation))
    losses.append(loss)
    dw= (activation - y) *x
    db= activation - y
    w= w-learning_rate*np.mean(dw)
    b= b-learning_rate*np.mean(db)
  return w,b,losses, initial_weights

def decision_boundary(w,b):

   X=symbols('X')
   equation_1= Eq( X*w+b,0)
   solution= solve(equation_1,X)
   return solution

#example_set:

#Inputs:
x = np.array([1,2,3,4,5])
# Labels
y = np.array([0, 0, 1, 1, 1])

w,b,losses,initial_weights= train(x,y,learning_rate=0.1, epochs=100)
print(f"initial weights: {initial_weights}")
print(f"updated weight: {w}, updated bias:{b}")
print(f"initial loss: {losses[0]}, final loss:{losses[-1]}")
solution=decision_boundary(w,b)
print(f"decision boundary:{solution}")
plt.scatter(x[y==0],y[y==0], color='green',label='class 0')
plt.scatter(x[y==1],y[y==1], color='red', label='class 1')
x_values= np.linspace(0,10,100)
y_values= sigmoid(x_values*w+b)
plt.plot(x_values,y_values,color='blue',label='Decision Boundary')
plt.xlabel("x")
plt.ylabel("probabilities")
plt.legend()
plt.show()

plt.plot(losses)
plt.xlabel("epochs")
plt.ylabel("losses")
plt.show()

