import numpy as np
from functions import softmax, cross_entropy_error

class Relu:
    def __init__(self):
        self.mask = None
        
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx
    

class Sigmoid:
    def __init__(self):
        self.out = None
        
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        
        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        
        return dx
    
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        return dx
    
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        
        return dx
 
'''
# 테스트 코드   
if __name__ == "__main__":
    np.random.seed(0)

    # 입력 (2개 샘플, 3차원)
    x = np.random.randn(2, 3)
    W = np.random.randn(3, 2)
    b = np.random.randn(2)
    t = np.array([[0, 1], [1, 0]])  # 원-핫 레이블

    print("=== 순전파 ===")
    affine = Affine(W, b)
    sigmoid = Sigmoid()
    relu = Relu()
    last_layer = SoftmaxWithLoss()

    out_affine = affine.forward(x)
    print("Affine 출력:", out_affine)

    out_sigmoid = sigmoid.forward(out_affine)
    print("Sigmoid 출력:", out_sigmoid)

    out_relu = relu.forward(out_sigmoid)
    print("ReLU 출력:", out_relu)

    loss = last_layer.forward(out_relu, t)
    print("Loss:", loss)

    print("\n=== 역전파 ===")
    dout = last_layer.backward()
    print("SoftmaxWithLoss backward:", dout)

    d_relu = relu.backward(dout)
    d_sigmoid = sigmoid.backward(d_relu)
    dx = affine.backward(d_sigmoid)

    print("Affine dW:", affine.dW)
    print("Affine db:", affine.db)
    print("입력 방향 dx:", dx)
'''
