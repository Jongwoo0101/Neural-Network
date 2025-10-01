import numpy as np
import matplotlib.pyplot as plt

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val  # 원래 값 복원

    return grad

# 주어진 함수 f(x0, x1) = x0 + 1/x1
def f(x):
    return x[0] + 1.0/x[1]

# 점 (1,2)
x = np.array([1.0, 2.0])
# (1,2)에서의 gradient
x0 = np.array([1.0, 2.0])
grad = numerical_gradient(f, x0)


# 그래프 영역 설정
x = np.linspace(-1, 3, 50)
y = np.linspace(0.5, 3, 50)   # x1=0은 분모가 0되므로 피함
X, Y = np.meshgrid(x, y)
Z = X + 1.0/Y

# 3D 그래프
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')

# 함수 곡면
ax.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')

# 점 (1,2,f(1,2))
z0 = f(x0)
ax.scatter(x0[0], x0[1], z0, color='r', s=50, label='Point (1,2)')

# Gradient 벡터를 화살표로 표시 (2D 평면 위에서)
ax.quiver(
    x0[0], x0[1], z0,      # 시작점
    grad[0], grad[1], 0,   # x, y 방향 성분 (z방향은 0)
    length=0.5, color='r', linewidth=2
)

ax.set_xlabel('x0')
ax.set_ylabel('x1')
ax.set_zlabel('f(x0, x1)')
ax.set_title('f(x0, x1) = x0 + 1/x1 with Gradient at (1,2)')
ax.legend()
plt.show()

print(grad)