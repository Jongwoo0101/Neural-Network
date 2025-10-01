import numpy as np

t = np.array([1, 0, 0])

y = np.array([2.0, 0.1, 0.2])

def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

def softmax(a):
    exp_a = np.exp(a - np.max(a)) 
    return exp_a / np.sum(exp_a)

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


mse = mean_squared_error(y, t)
print("MSE:", mse)

# Softmax 적용해야 정상
y_prob = softmax(y)

cross_entropy = cross_entropy_error(y_prob, t)
print("Cross Entropy:", cross_entropy)
