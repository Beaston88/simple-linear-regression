import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the datasets

X = pd.read_csv('linearX.csv', header=None).values.flatten()
Y = pd.read_csv('linearY.csv', header=None).values.flatten()

# Display basic information about the datasets

print("First five rows of X (Predictor Variable):", X[:5])
print("First five rows of Y (Response Variable):", Y[:5])
print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)

First five rows of X (Predictor Variable): [9.1 8. 9.1 8.4 6.9]
First five rows of Y (Response Variable): [0.99523 0.99007 0.99769
0.99386 0.99508]
Shape of X: (100,)
Shape of Y: (100,)

#Normalizing the Data

X_normalized = (X - np.mean(X)) / np.std(X)
print("First five rows of normalized X:", X_normalized[:5])
First five rows of normalized X: [ 0.60239429 -0.03598116 0.60239429
0.19615537 -0.67435661]

#Implementing Simple Linear Regression with Gradient Descent

def compute_cost(X, Y, theta):
m = len(Y)
predictions = X.dot(theta)
cost = (1/(2*m)) * np.sum((predictions - Y)**2)
return cost
def gradient_descent(X, Y, theta, learning_rate, iterations):
m = len(Y)
cost_history = np.zeros(iterations)
for it in range(iterations):
prediction = X.dot(theta)
theta -= (1/m) * learning_rate * (X.T.dot(prediction - Y))
cost_history[it] = compute_cost(X, Y, theta)
return theta, cost_history
X_b = np.c_[np.ones((X_normalized.shape[0], 1)), X_normalized]
theta = np.zeros(X_b.shape[1])
learning_rate = 0.5
iterations = 1000
theta_final, cost_history = gradient_descent(X_b, Y, theta,
learning_rate, iterations)
print("Theta after convergence:", theta_final)
print("Final cost:", cost_history[-1])
Theta after convergence: [0.9966201 0.0013402]
Final cost: 1.1947898109836605e-06

#Plotting Cost Function vs Iterations

plt.plot(range(1, 51), cost_history[:50], 'b-')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function vs. Iterations (First 50 Iterations)')
plt.show()

#Plotting the Dataset and the Fitted Line

plt.scatter(X_normalized, Y, color='blue', label='Data points')
plt.plot(X_normalized, X_b.dot(theta_final), color='red',
label='Fitted line')
plt.xlabel('Normalized X')
plt.ylabel('Y')
plt.title('Dataset and Fitted Line')
plt.legend()
plt.show()

#Experimenting with Different Learning Rates

learning_rates = [0.005, 0.5, 5]
cost_histories = {}
for lr in learning_rates:
_, cost_histories[lr] = gradient_descent(X_b, Y, theta, lr,
iterations)
plt.plot(range(1, 51), cost_histories[lr][:50], label=f'lr={lr}')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function vs. Iterations for Different Learning Rates')
plt.legend()
plt.show()

#Implementing Stochastic and Mini-batch Gradient Descent

def stochastic_gradient_descent(X, Y, theta, learning_rate,
iterations):
m = len(Y)
cost_history = np.zeros(iterations)
for it in range(iterations):
for i in range(m):
rand_index = np.random.randint(0, m)
X_i = X[rand_index, :].reshape(1, -1)
Y_i = Y[rand_index].reshape(1, )
prediction = np.dot(X_i, theta) gradient = X_i.T.dot(prediction - Y_i)
theta -= (1/m) * learning_rate * gradient.flatten()
cost_history[it] = compute_cost(X, Y, theta)
return theta, cost_history
def mini_batch_gradient_descent(X, Y, theta, learning_rate,
iterations, batch_size):
m = len(Y)
cost_history = np.zeros(iterations)
for it in range(iterations):
indices = np.random.permutation(m)
X = X[indices]
Y = Y[indices]
for i in range(0, m, batch_size):
X_i = X[i:i+batch_size]
Y_i = Y[i:i+batch_size]
prediction = np.dot(X_i, theta)
theta -= (1/m) * learning_rate * (X_i.T.dot(prediction -
Y_i))
cost_history[it] = compute_cost(X, Y, theta)
return theta, cost_history
theta_sgd, cost_history_sgd = stochastic_gradient_descent(X_b, Y,
theta, learning_rate, iterations)
theta_mbgd, cost_history_mbgd = mini_batch_gradient_descent(X_b, Y,
theta, learning_rate, iterations, batch_size=20)
plt.plot(range(1, 51), cost_history[:50], 'b-', label='Batch GD')
plt.plot(range(1, 51), cost_history_sgd[:50], 'green',
label='Stochastic GD')
plt.plot(range(1, 51), cost_history_mbgd[:50], 'red', label='Mini
batch GD')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function vs. Iterations for Different GD Methods')
plt.legend()
plt.show()
