import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.util import shuffle_dataset
from common.trainer import Trainer

# MNIST 데이터 불러오기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 학습 데이터 축소 (빠른 테스트용)
x_train = x_train[:500]
t_train = t_train[:500]

# 20%를 검증 데이터로 분할
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)
x_train, t_train = shuffle_dataset(x_train, t_train)
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]


# 학습 함수 정의
def __train(lr, weight_decay, epochs=50):
    network = MultiLayerNet(input_size=784,
                            hidden_size_list=[100, 100, 100, 100, 100, 100],
                            output_size=10,
                            weight_decay_lambda=weight_decay)
    trainer = Trainer(network, x_train, t_train, x_val, t_val,
                      epochs=epochs,
                      mini_batch_size=100,
                      optimizer='sgd',
                      optimizer_param={'lr': lr},
                      verbose=False)
    trainer.train()
    return trainer.test_acc_list, trainer.train_acc_list


# 좁힌 범위에서 하이퍼파라미터 무작위 탐색
# 이전 탐색 결과 기반 범위 설정 예시
lr_min, lr_max = 0.0005, 0.002
weight_decay_min, weight_decay_max = 1e-7, 5e-6

optimization_trial = 50  # 횟수
results_val = {}
results_train = {}

for _ in range(optimization_trial):
    lr = np.random.uniform(lr_min, lr_max)
    weight_decay = np.random.uniform(weight_decay_min, weight_decay_max)

    val_acc_list, train_acc_list = __train(lr, weight_decay)

    key = f"lr:{lr:.6f}, weight_decay:{weight_decay:.8f}"
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list

    print(f"val acc: {val_acc_list[-1]:.4f} | lr: {lr:.6f}, weight_decay: {weight_decay:.8f}")

# 최적 파라미터 확인
best_key = max(results_val, key=lambda k: results_val[k][-1])
best_val_acc = results_val[best_key][-1]
print("\n=========== 최적 하이퍼파라미터 ===========")
print("최적 파라미터:", best_key)
print("최고 검증 정확도:", best_val_acc)

# 상위 20개 결과 시각화
graph_draw_num = 20
col_num = 5
row_num = int(np.ceil(graph_draw_num / col_num))
i = 0

plt.figure(figsize=(12, 8))

for key, val_acc_list in sorted(results_val.items(), key=lambda x: x[1][-1], reverse=True):
    plt.subplot(row_num, col_num, i + 1)
    plt.title(f"Best-{i + 1}")
    plt.ylim(0.0, 1.0)
    if i % 5: plt.yticks([])
    plt.xticks([])
    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list)
    plt.plot(x, results_train[key], "--")
    i += 1
    if i >= graph_draw_num:
        break

plt.subplots_adjust(wspace=0.2, hspace=0.3)
plt.show()
