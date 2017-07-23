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

# network = init_network()
# x = np.array([0.1, 0.5])
# y = forward(network, x)
# print(y)

# 3.5 出力層の設計
# 3.5.1 恒等関数とソフトマックス関数
# def softmax(a):
#     exp_a = np.exp(a)
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a
#
#     return y

# 3.5.2 ソフトマックス関数の実装上の注意
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

# 3.5.3 ソフトマックス関数の特徴

# 3.5.4出力層のニューロンの数

# 3.6 手書き数字認識
# 3.6.1 MNISTデータセット
import sys, os
import pickle
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image

# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# print(x_train.shape)
# print(t_train.shape)
# print(x_test.shape)
# print(t_test.shape)

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
#
# img = x_test[0]
# label = t_train[0]
#
# print(label)
#
# print(img.shape)
# img = img.reshape(28, 28)
# print(img.shape)
#
# img_show(img)

# 3.6.2 ニューラルネットワークの推論処理
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

# x, t = get_data()
# network = init_network()
#
# accuracy_cnt = 0
# for i in range(len(x)):
#     y = predict(network, x[i])
#     p = np.argmax(y)
#
#     if p == t[i]:
#         accuracy_cnt += 1
#
# print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

# 3.6.3 バッチ処理
# x, t = get_data()
# network = init_network()
#
# batch_size = 100
# accuracy_cnt = 0
#
# for i in range(0, len(x), batch_size):
#     x_batch = x[i:i+batch_size]
#     y_batch = predict(network, x_batch)
#     p = np.argmax(y_batch, axis=1)
#
#     accuracy_cnt += np.sum(p == t[i:i+batch_size])

# print("Accuracy:" + str(float(accuracy_cnt) / len(x)))


# 4章 ニューラルネットワークの学習
# 4.1 データから学習する

# 4.1.1 データ駆動

# 4.1.2 訓練データとテストデータ

# 4.2 損失関数

# 4.2.1 2乗和誤差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# # y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
# y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
# print(mean_squared_error(np.array(t), np.array(y)))

# 4.2.2 交差エントロピー誤差
# def cross_entropy_error(y, t):
#     delta = 0.0000001
#     return -np.sum(t * np.log(y + delta))

# t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# # y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
# y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
# print(cross_entropy_error(np.array(y), np.array(t)))
#
# # 4.2.3 ミニバッチ学習
# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# print(x_train.shape)
# print(t_train.shape)
#
# train_size = x_train.shape[0]
# batch_size = 10
# batch_mask = np.random.choice(train_size, batch_mask)
# x_batch = x_train[batch_mask]
# t_batch = t_train[batch_mask]

# 4.2.4 [バッチ対応版]交差エントロピー誤差の実装
# def cross_entropy_error(y, t):
#     if y.ndim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)
#
#     batch_size = y.shape[0]
#     return -np.sum(t * np.log(y)) / batch_size

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

# 4.2.5 なぜ損失関数を設定するのか？

# 4.3 数値微分
# 4.3.1 微分
def numerical_diff(f, x):
    h = 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

# 4.3.2 数値微分の例
def function_1(x):
    return 0.01*x**2 + 0.1*x

# x = np.arange(0.0, 20.0, 0.1)
# y = function_1(x)
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.plot(x, y)
# plt.show()

# 4.3.3 偏微分
def function_2(x):
    return x[0]**2 + x[1]**2

# 4.4 勾配
def numerical_gradient(f, x):
    h = 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad

# print(numerical_gradient(function_2, np.array([3.0, 4.0])))
# print(numerical_gradient(function_2, np.array([0.0, 2.0])))
# print(numerical_gradient(function_2, np.array([3.0, 0.0])))

# 4.4.1 勾配法
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x

# init_x = np.array([-3.0, 4.0])
# print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))

# 4.4.2 ニューラルネットワークに対する勾配
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
print(np.argmax(p))
t = np.array([0, 0, 1])
print(net.loss(x, t))

# def f(W):
#     return net.loss(x, t)

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)
print(dW)
