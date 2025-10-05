import numpy as np
from functions import softmax, sigmoid

x = np.array([1, 2]) # x1, x2 = 1, 2

b1 = np.array([1, 1]) # bias1 = 1, 1
b2 = np.array([1, -1]) # bias2 = 1, -1
b3 = np.array([-1, 1]) # bias3 = -1, 1

w1 = np.array([[2, -1], [1, 1]])
w2 = np.array([[0, 2], [1, 1]])
w3 = np.array([[1, 2], [-2, 1]])

# 3번 문제
a1 = np.dot(x, w1) + b1
z1 = sigmoid(a1)
print(z1)
 
# 4번 문제
a2 = np.dot(z1, w2) + b2
z2 = sigmoid(a2)
print(z2)

# 5번 문제
a3 = np.dot(z2, w3) + b3
print(softmax(a3))