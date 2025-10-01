import numpy as np
import matplotlib.pyplot as plt

# 계단 함수
def step_function(x):
    return np.array(x > 0, dtype=int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)

plt.title("step function")
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# 시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.title("sigmoid function")
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# 렐루 함수
def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)

plt.title("relu function")
plt.plot(x, y)
plt.ylim(-1, 5)
plt.show()
