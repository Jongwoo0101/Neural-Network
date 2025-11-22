import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import Adam   # Adam 사용


# 0. MNIST 데이터 ==========
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000


# 1. 실험 설정 ==========
weight_init_types = {'Xavier': 'sigmoid', 'He': 'relu'}

optimizer = Adam(lr=0.001)  # Xavier, He 둘 다 잘 수렴하는 최적 learning rate

networks = {}
train_loss = {}

for key, weight_type in weight_init_types.items():
    networks[key] = MultiLayerNet(
        input_size=784,
        hidden_size_list=[100, 100, 100, 100],
        output_size=10,
        weight_init_std=weight_type
    )
    train_loss[key] = []


# 2. 훈련 시작 ==========
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in weight_init_types.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizer.update(networks[key].params, grads)

        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)

    if i % 100 == 0:
        print("========== iteration:", i, "==========")
        for key in weight_init_types.keys():
            print(key, ":", networks[key].loss(x_batch, t_batch))


# 3. 손실 그래프 ==========
markers = {'Xavier': 's', 'He': 'D'}
x = np.arange(max_iterations)

for key in weight_init_types.keys():
    plt.plot(x, smooth_curve(train_loss[key]),
             marker=markers[key], markevery=100, label=key)

plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 2.5)
plt.legend()
plt.show()
