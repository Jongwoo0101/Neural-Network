import sys, os
import numpy as np
import pickle
sys.path.append(os.path.abspath('data'))
from data.mnist import load_mnist
from functions import softmax, sigmoid, get_data, init_network, predict

x, t = get_data()
network = init_network()

# 전체 정확도 카운트
accuracy_cnt = 0

# 숫자별 카운트
num_classes = 10
correct_per_class = np.zeros(num_classes, dtype=np.int32)
total_per_class = np.zeros(num_classes, dtype=np.int32)

for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y) 
    total_per_class[t[i]] += 1
    if p == t[i]:
        accuracy_cnt += 1
        correct_per_class[t[i]] += 1

# 전체 정확도
print("전체 Accuracy:", float(accuracy_cnt) / len(x))

# 숫자별 정확도
for i in range(num_classes):
    acc = correct_per_class[i] / total_per_class[i]
    print(f"숫자 {i} 정확도: {acc:.4f}")

# 숫자 8 정확도
print("\n숫자 8의 정확도:", correct_per_class[8] / total_per_class[8])

# 가장 정확도가 낮은 숫자
class_acc = correct_per_class / total_per_class
worst_class = np.argmin(class_acc)
print(f"\n가장 인식 잘 못하는 숫자: {worst_class} (정확도 {class_acc[worst_class]:.4f})")
