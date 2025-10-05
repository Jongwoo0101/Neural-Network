import sys, os
sys.path.append(os.pardir)
import numpy as np
from functions import softmax, cross_entropy_error, numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) # 정규분포로 초기화
        
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        
        return loss
    
    def cross_entropy_error(y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
            
        batch_size = y.shape[0]
        return -np.sum(t * np.log(y + 1e-7)) / batch_size

'''
# test 
net = simpleNet()
print(net.W) # 가중치 매개변수

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
print(np.argmax(p)) # 최댓값의 인덱스

t = np.array([0, 0, 1]) # 정답 레이블
print(net.loss(x, t))
'''