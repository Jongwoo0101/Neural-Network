import sys, os
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from optimizer import *

# 목적 함수
def f(x, y):
    return (1/10) * x**2 + y**2

# 편미분
def df(x, y):
    return (1/5) * x, 2 * y

# 초기값
init_pos = (-5.0, 1.0)
params = {'x': init_pos[0], 'y': init_pos[1]}
grads = {'x': 0, 'y': 0}

# Optimizer 설정
optimizers = OrderedDict()
optimizers["SGD"] = SGD(lr=0.95)
optimizers["Momentum"] = Momentum(lr=0.1)
optimizers["AdaGrad"] = AdaGrad(lr=1.5)
optimizers["Adam"] = Adam(lr=0.3)

plt.figure(figsize=(12, 8))
idx = 1

# 각 optimizer에 대해 반복
for key in optimizers:
    optimizer = optimizers[key]
    x_history, y_history = [], []
    params['x'], params['y'] = init_pos[0], init_pos[1]
    
    for i in range(50):
        x_history.append(params['x'])
        y_history.append(params['y'])
        
        grads['x'], grads['y'] = df(params['x'], params['y'])
        optimizer.update(params, grads)
    
    # 출력
    print(f"[{key}]")
    print("x_history:", x_history)
    print("y_history:", y_history)
    print()
    
    # 등고선 그래프
    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    
    mask = Z > 7
    Z[mask] = 0
    
    plt.subplot(2, 2, idx)
    idx += 1
    plt.plot(x_history, y_history, 'o-', color="red")
    plt.contour(X, Y, Z)
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.plot(0, 0, '+')
    plt.title(key)
    plt.xlabel("x")
    plt.ylabel("y")

plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.show()
