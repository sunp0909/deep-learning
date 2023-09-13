import numpy as np
import matplotlib.pyplot as plt

def num_diff(f,x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2 * h)

x = np.arange(0,20,0.01)

def function_1(x):
    return 0.01 * x**2 + 0.1 * x

y = function_1(x)
y_ = num_diff(function_1,x)

# print(y_)

# plt.plot(x, y)
# plt.plot(x,y_)

# plt.show()

def function_2(x):
    return x[0] ** 2 + x[1] ** 2

def num_grad(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    for i in range(x.size):
        tmp_val = x[i]
        
        x[i] = tmp_val + h
        fxh1 = f(x)
        
        x[i] = tmp_val - h
        fxh2 = f(x)
        
        grad[i] = (fxh1 - fxh2) / (2 * h)
        x[i] = tmp_val
    
    return grad

# print(num_grad(function_2, np.array([3.0, 4.0])))
# print(num_grad(function_2, np.array([0.0, 2.0])))
# print(num_grad(function_2, np.array([3.0, 0.0])))
