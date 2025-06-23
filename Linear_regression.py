import numpy as np
import matplotlib.pyplot as plt

def initialization():
  w=np.random.rand()
  b=np.random.rand()
  return w, b
  
def activation(w,b,x):
  return x*w+b
  
def loss_function(y_target,y_pred):
  loss=(y_target-y_pred)**2
  return loss
  
def gradient(w,b,x,y_target):
  y_pred= activation(w,b,x)
  dw=-2*(y_target-y_pred)*x
  db=-2*(y_target-y_pred)
  return dw, db
  
def update_weights(w,b,dw,db, learning_rate):
  w= w - learning_rate*dw
  b= b -  learning_rate*db
  return w, b
  
def gradient_descent(x, y_target, epochs=100, learning_rate= 0.01):
  w, b =initialization()
  initial_weights= (w,b)
  losses=[]
  for i in range (epochs):
    y_pred= activation(w,b,x)
    loss= loss_function(y_target,y_pred)
    losses.append(loss)
    dw, db= gradient(w,b,x,y_target)
    w, b = update_weights(w,b,dw,db, learning_rate)

  final_loss= loss_function(y_target,activation(w,b,x))
  return w,b,losses, final_loss, initial_weights

#example_set
x=2.0
y_target= 5.0
w, b,losses,final_loss, initial_weights = gradient_descent(x, y_target, epochs=100, learning_rate=0.01)

print(f"initial loss is:{losses[0]}, final loss is:{final_loss}")
print(f"our initial weights are:{initial_weights}")
print(f"final_w: {w}, final_b: {b}")

plt.plot(losses, label="loss over epoches")
plt.legend()

plt.xlabel("epochs")
plt.ylabel("losses")
plt.show()
