import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.functions import numerical_gradient


def function_f(x):
    return 1 + 2 * x[1] + x[0] * x[1] * x[2] + np.power(x[2], 3.0)

x = np.array([0.0, 1.0, 2.0])
print(numerical_gradient(function_f, x))