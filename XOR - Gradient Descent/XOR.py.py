import numpy as np
np.random.seed(42)
# Neural Network for XOR Problem
# 0 0 -> 0
# 0 1 -> 1
# 1 0 -> 1
# 1 1 -> 0

#Activation function used: Sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x)*(1-sigmoid(x))

def forward(x, w1, w2, predict = False):
    a1 = np.matmul(x,w1)
    z1 = sigmoid(a1)
    # print("Z1: ",z1)

    bias = np.ones((len(z1), 1))
    z1 = np.concatenate((bias, z1), axis = 1)
    a2 = np.matmul(z1,w2)
    z2 = sigmoid(a2)
    if predict:
        return z2

    return a1, z1, a2, z2

def backprop(a2, z0, z1, z2, y):
    delta2 = z2 - y
    Delta2 = np.matmul(z1.T, delta2)

    delta1 = (delta2.dot(w2[1:,:].T))* sigmoid_deriv(a1)
    Delta1 = np.matmul(z0.T, delta1)

    return delta2, Delta1, Delta2

# First column is the bias
X = np.array([[1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])

print(X.shape)

y = np.array([[0],[1],[1],[0]])

# print(X, y)

w1 = np.random.randn(3,1)
print(w1.shape)
w2 = np.random.randn(2,1)
print(w2)

lr = 0.01

costs = []
epochs = 50000

m = len(X)

for i in range(epochs):

    a1, z1, a2, z2 = forward(X, w1, w2)

    delta2, Delta1, Delta2 = backprop(a2, X, z1, z2, y)

    w1 -= lr*(1/m)*Delta1
    w2 -= lr*(1/m)*Delta2

    c = np.mean(np.abs(delta2))
    costs.append(c)

    if i % 1000 == 0:
        print("Iteration: {}. Error: {}".format(i,c))

print("Model Trained.")

z3 = forward(X,w1,w2,True)
print("Predictions: ")
print(np.round(z3))
print("Weights: ")
print(w1[1:])

