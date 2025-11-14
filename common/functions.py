# 자주 사용하는 함수들 모음
# 코드 재사용 방지, 메모리 최적화 시도용

import numpy as np
import sys, os
import numpy as np
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.mnist import load_mnist
from PIL import Image

def softmax(a):
    c = np.max(a)  # 안정화
    exp_a = np.exp(a - c)
    return exp_a / np.sum(exp_a)

def sigmoid(x):
    # overflow 방지 버전
    pos_mask = (x >= 0)
    neg_mask = ~pos_mask
    z = np.zeros_like(x, dtype=np.float64)
    z[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[neg_mask])
    z[neg_mask] = exp_x / (1 + exp_x)
    return z

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten = True, normalize = True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open(os.path.join("4주차", "sample_weight.pkl"), 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    
    return y

def sigmoid(x):
    # overflow 방지 버전
    pos_mask = (x >= 0)
    neg_mask = ~pos_mask
    z = np.zeros_like(x, dtype=np.float64)
    z[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[neg_mask])
    z[neg_mask] = exp_x / (1 + exp_x)
    return z

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


# 다차원 배열 탐색 가능 버전
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    # 다차원 배열 대응을 위해 np.ndindex 사용
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]

        # f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 값 복원
        it.iternext()

    return grad



# def numerical_gradient(f, x):
#     h = 1e-4
#     grad = np.zeros_like(x)
    
#     for idx in range(x.size):
#         tmp_val = x[idx]
#         x[idx] = tmp_val +h
#         fxh1 = f(x)
        
#         x[idx] = tmp_val - h
#         fxh2 = f(x)
        
#         grad[idx] = (fxh1 - fxh2) / (2 * h)
#         x[idx] = tmp_val
#     return grad

