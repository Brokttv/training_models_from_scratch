import numpy as np
import matplotlib.pyplot as plt

def initialization():
     w_input = np.array([[0.2, -0.3, 0.6], [-0.5, 0.7, -0.1], [0.8, -0.2, 0.9], [1.0, 0.4, -0.7]]) # Shape: (hidden_neurons, input_features)
    b_input = np.array([0.1, -0.2, 0.3, 0.5])  # Shape: (hidden_neurons,)
    w_output = np.array([[0.7, -0.2, 0.1, 0.3], [-0.3, 0.8, -0.4, 0.2], [0.5, -0.6, 0.9, -0.7]]) # Shape: (output_neurons, hidden_neurons)
    b_output = np.array([0.1, -0.2, 0.05])  # Shape: (output_neurons,)
    return w_input, w_output, b_input, b_output

def Relu(x):
    return np.maximum(0, x)

def Relu_derivative(z):
    return np.where(z > 0, 1, 0)

def forward_pass(x, w_input, w_output, b_input, b_output):
    z_input = np.dot(x, w_input.T) + b_input  # Shape: (num_samples, hidden_neurons)  
    z_input_activation = Relu(z_input)        # Shape: (num_samples, hidden_neurons)  
    z_output = np.dot(z_input_activation, w_output.T) + b_output  # Shape: (num_samples, output_neurons)  
    return z_input_activation, z_output, z_input

def loss_function(y, z_output):
    e = y - z_output  # Shape: (num_samples, output_neurons)  
    loss = np.mean(e**2)  # Scalar
    return loss, e

def backward_pass(w_input, w_output, b_output, b_input, x, e, z_output, z_input, z_input_activation, lr=0.01):
    num_samples = x.shape[0]

    # Gradients for output layer
    deltas_output = -2 * e / num_samples  # Shape: (num_samples, output_neurons)  # Shape: (num_samples, output_neurons)
    dw_output = np.dot(z_input_activation.T, deltas_output) / num_samples  # Shape: (hidden_neurons, output_neurons)  # Shape: (hidden_neurons, output_neurons)
    db_output = np.sum(deltas_output, axis=0)/ num_samples  # Shape: (output_neurons,) # Shape: (output_neurons,)

    # Gradients for hidden layer
    activation_prime =  Relu_derivative(z_input)  # Shape: (num_samples, hidden_neurons)  
    deltas_input = np.dot(deltas_output, w_output) * activation_prime / num_samples # Shape: (num_samples, hidden_neurons)
    dw_input = np.dot(x.T, deltas_input) / num_samples # Shape: (input_features, hidden_neurons)
    db_input = np.sum(deltas_input, axis=0) / num_samples  # Shape: (hidden_neurons,)

    # Update weights and biases
    w_input -= lr * dw_input.T
    w_output -= lr * dw_output.T
    b_input -= lr * db_input
    b_output -= lr * db_output

    return w_input, w_output, b_input, b_output


def batch_gradient_descent(x, y, lr=0.01, epochs=100):
    w_input, w_output, b_input, b_output = initialization()
    losses = []
    for i in range(epochs):

        z_input_activation, z_output, z_input = forward_pass(x, w_input, w_output, b_input, b_output)
        loss, e = loss_function(y, z_output)
        losses.append(loss)
        w_input, w_output, b_input, b_output = backward_pass(w_input, w_output, b_output, b_input, x, e, z_output, z_input, z_input_activation, lr)
    return w_input, w_output, b_input, b_output, losses

# Test set
x = np.array([[1.0, 2.0, 6.0],
              [2.0, 3.0, 5.0],
              [3.0, 4.0, 8.0],
              [4.0, 5.0, 7.0],
              [5.0, 6.0, 9.0],
              [6.0, 7.0, 10.0],
              [7.0, 8.0, 11.0],
              [8.0, 9.0, 12.0],
              [9.0, 10.0, 13.0],
              [10.0, 11.0, 14.0]])  # Input features (3 samples, 3 features)

y = np.array([[3.0, 5.0, 7.0],
              [4.0, 6.0, 8.0],
              [5.0, 7.0, 9.0],
              [6.0, 8.0, 10.0],
              [7.0, 9.0, 11.0],
              [8.0, 10.0, 12.0],
              [9.0, 11.0, 13.0],
              [10.0, 12.0, 14.0],
              [11.0, 13.0, 15.0],
              [12.0, 14.0, 16.0]])  # Target values (3 samples, 3 output neurons)

# Train the model

w_input, w_output, b_input, b_output, losses = gradient_descent(x, y, lr=0.01, epochs=100)

# Print initial and final loss
print("Initial loss:", losses[0])
print("Final loss:", losses[-1])

# Plot the loss curve
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training Loss Curve")
plt.show()
