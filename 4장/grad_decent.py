from diff import *
import matplotlib.pyplot as plt

def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x
    result = []
    for i in range(step_num):
        result.append(x.copy())
        grad = num_grad(f, x)
        x -= lr * grad

    return x,np.array(result)

init_x = np.array([-3.0, 4.0])
# gradient_descent(function_2,init_x, lr=0.01, step_num=10)
# print(gradient_descent(function_2,init_x, lr=0.1, step_num=100))


# 그래프
init_x = np.array([-3.0, 4.0])
x, x_history = gradient_descent(function_2, init_x, lr=0.1, step_num=20)

plt.plot([-5, 5], [0, 0], '--b')
plt.plot([0, 0], [-5, 5], '--b')
plt.plot(x_history[:, 0], x_history[:, 1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()