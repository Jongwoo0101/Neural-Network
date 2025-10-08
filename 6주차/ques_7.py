import numpy as np
import matplotlib.pyplot as plt
import time, sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 부모 디렉터리의 파일을 가져올 수 있도록 설정

from data.mnist import load_mnist
from two_layer_net import TwoLayerNet

start_time = time.time()

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 데이터 크기를 1/4로 축소
train_size = x_train.shape[0] // 4
test_size = x_test.shape[0] // 4

x_train = x_train[:train_size]
t_train = t_train[:train_size]
x_test = x_test[:test_size]
t_test = t_test[:test_size]

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 하이퍼파라미터
iters_num = 1000
train_size = x_train.shape[0]
batch_size = 80
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []
epoch_list = []   # 정확도 계산 시점 기록용

# 1에포크당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산 (numerical_gradient)너무 느려서 gradient로 대체
    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1에포크당 정확도 계산
    # if i % iter_per_epoch == 0:
    #     train_acc = network.accuracy(x_train, t_train)
    #     test_acc = network.accuracy(x_test, t_test)
    #     train_acc_list.append(train_acc)
    #     test_acc_list.append(test_acc)
    #     print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
    # 추가 문제 20번 반복마다 정확도 계산
    if i % 20 == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        epoch_list.append(i)
        print(f"iter {i:4d} | train acc: {train_acc:.4f}, test acc: {test_acc:.4f}")

x = np.arange(len(train_acc_list))  # 에포크 수

# 7번문제
# plt.figure(figsize=(8, 5))
# plt.plot(x, train_acc_list, label='Training accuracy')
# plt.plot(x, test_acc_list, label='Test accuracy', linestyle='--')
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.title("Training vs Test Accuracy")
# plt.legend()
# plt.grid(True)
# plt.show()

# 추가문제 20번 반복마다 정확도 시각화
# plt.figure(figsize=(8, 5))
# plt.plot(epoch_list, train_acc_list, label='Training accuracy')
# plt.plot(epoch_list, test_acc_list, label='Test accuracy', linestyle='--')
# plt.xlabel("Iteration")
# plt.ylabel("Accuracy")
# plt.title("Training vs Test Accuracy (every 20 iterations)")
# plt.legend()
# plt.grid(True)
# plt.show()

# 추가문제 loss값 추이 그래프
plt.figure(figsize=(8, 5))
plt.plot(np.arange(len(train_loss_list)), train_loss_list, color='tab:blue')
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss over Iterations")
plt.grid(True)
plt.show()

end_time = time.time()

print(end_time - start_time)