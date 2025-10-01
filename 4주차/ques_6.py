import sys, os
import numpy as np
import pickle
sys.path.append(os.path.abspath('data'))
from data.mnist import load_mnist
from PIL import Image
from functions import softmax, sigmoid, get_data, init_network, predict

x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y) 
    if p == t[i]:
        accuracy_cnt += 1
        
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
    
