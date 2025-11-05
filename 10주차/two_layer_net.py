import numpy as np
from layers import *
from functions import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 신경망의 가중치와 편향을 저장할 딕셔너리 생성
        self.params = {}

        # 1층(입력층 → 은닉층)의 가중치 W1 초기화
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        # 1층의 편향 b1을 0으로 초기화
        self.params['b1'] = np.zeros(hidden_size)

        # 2층(은닉층 → 출력층)의 가중치 W2 초기화
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        # 2층의 편향 b2를 0으로 초기화
        self.params['b2'] = np.zeros(output_size)

        # 순서가 보장된 계층 저장용 OrderedDict 생성
        self.layers = OrderedDict()
        # 첫 번째 Affine(전결합) 계층 생성
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        # 활성화 함수 계층 (ReLU)
        self.layers['Relu1'] = Relu()
        # 두 번째 Affine 계층 (은닉층 → 출력층)
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        # 마지막 층: 소프트맥스 + 손실 함수(교차 엔트로피)
        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        # 입력 데이터를 받아 각 계층을 순서대로 통과시키며 출력 계산
        for layer in self.layers.values():
            x = layer.forward(x)   # forward() 함수를 통해 Affine → ReLU → Affine 순으로 진행
        return x                   # 최종 출력(로짓 값, 즉 Softmax 이전 값)을 반환

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
    
    def gradient(self, x, t):
        # 순전파를 통해 손실 계산 (SoftmaxWithLoss 계층의 forward 호출)
        self.loss(x, t)

        # 역전파의 시작점: 손실에 대한 미분값은 항상 1
        dout = 1
        # SoftmaxWithLoss 계층의 역전파 수행 (예측과 정답 간의 오차 전파)
        dout = self.lastLayer.backward(dout)

        # 순전파 때의 층 순서를 뒤집어 역전파 순서대로 처리
        layers = list(self.layers.values())
        layers.reverse()

        # 각 층에서 backward() 실행하여 미분값을 역전파
        for layer in layers:
            dout = layer.backward(dout)

        # 각 층의 기울기(가중치 W, 편향 b)를 grads 딕셔너리에 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        # 계산된 모든 기울기 반환
        return grads
