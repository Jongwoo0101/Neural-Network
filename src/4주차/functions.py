# 자주 사용하는 함수들 모음
# 코드 재사용 방지, 메모리 최적화 시도용

import sys, os
import numpy as np
import pickle
sys.path.append(os.path.abspath('data'))
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