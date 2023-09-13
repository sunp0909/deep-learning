# def step_func(x):
#     if x > 0:
#         return 1
#     else:
#         return 0

import numpy as np
    
def step_func(x):
    y = x > 0
    return y.astype(int)

a = np.array([1.0, 2.0])

# print(step_func(a))

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5.0, 5.0, 0.1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# plt.plot(x, sigmoid(x),linestyle = "--")
# plt.plot(x, step_func(x))
# plt.show()

def relu(x):
    return np.maximum(0,x)
# plt.plot(x,relu(x))
# plt.ylim(0,1)
# plt.show()

def softmax(x):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

b = np.array([1010, 1000, 990])

def identity_func(x):
    return x