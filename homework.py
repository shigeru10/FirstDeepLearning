# # 3.2.2 シグモイド関数
# def step_function(x):
#     if x > 0:
#         return 1
#     else:
#         return 0

# 3.2.3 ステップ関数のグラフ
import numpy as np
import matplotlib.pylab as plt

# def step_function(x):
#     return np.array(x > 0, dtype=np.int)
#
# x = np.arange(-5.0, 5.0, 0.1)
# y = step_function(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# plt.show()

# 3.2.4 シグモイド関数の実装
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# x = np.array([-1.0, 1.0, 2.0])
# print(sigmoid(x))

# x = np.arange(-5.0, 5.0, 0.1)
# y = sigmoid(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# plt.show()

# 3.2.7 ReLU関数
def relu(x):
    return np.maximum(0, x)

# 3.3.1 多次元配列

# 3.3.2 行列の積

# 3.3.3 ニューラルネットワークの行列の積
# X = np.array([1, 2])
# print(X.shape)
# W = np.array([[1, 3, 5], [2, 4, 6]])
# print(W.shape)
# Y = np.dot(X, W)
# print(Y)

# 3.4 3層ニューラルネットワークの実装
# 3.4.1 記号の確認

# 3.4.2 各層における信号伝達の実装
def identity_function(x):
    return x

# 3.4.3 実装まとめ
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

network = init_network()
x = np.array([0.1, 0.5])
y = forward(network, x)
print(y)
