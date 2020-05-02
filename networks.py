import random

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(-z))

class Networks(object):
    def __init__(self, sizes):
        """
        :type sizes: 各层神经元的数量 [2, 3, 1]: 第一层2个。第二层3个，第三层1个
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        # 激活函数
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SDG(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
         随机梯度下降算法
         :param training_data: 训练集，输入和对应期望输出 [[x1, y1],[x2, y2]...]
         :param pochs: 迭代期数量（训练多少个小批量数据集）
         :param mini_batch_size: 小批量数据集大小
         :param eta: 学习速率 （η）
         :param test_data: 测试数据
         :return
        """
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            # 随机排序
            random.shuffle(training_data)
            # 从训练集中按需分隔出多个小批量数据集
            mini_batches = [
                training_data[k: k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                # 对每个小数据集随机梯度下降
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("epoch {0} complete".format(j))
    def update_mini_batch(self, mini_batch, eta):
        """

        :param mini_batch:
        :param eta:
        :return:
        """
        # ▽b ▽w
        # np.zeros(b.shape) 给数据塞0
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            # △ ▽b， △ ▽w ,反向传播
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            # 调整w和b、
            """
            w = w - η/m * ∑
            b = b - η/m * ∑
            """
            self.weights = [
                w-(eta/len(mini_batch))*nw
                for w, nw in zip(self.weights, nabla_w)
            ]
            self.biases = [
                b-(eta/len(mini_batch))*nb
                for b, nb in zip(self.biases, nabla_b)
            ]

    def backprop(self, x, y):
        """
        反向传播算法
        :param x:
        :param y:
        :return:
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for i in range(2, self.num_layers):
            z = zs[-i]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-i+1].transpose(), delta) * sp
            nabla_b[-i] = delta
            nabla_w[-i] = np.dot(delta, activations[-i-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [
            (np.argmax(self.feedforward(x)), y)
            for (x, y) in test_data
        ]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

import mnist_loader
if __name__ == '__main__':
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    # 28*28个输入, 一个隐藏层30个神经元， 输入10个
    net = Networks([784, 30, 10])
    # 30次迭代期， 学习速率3.0, 小数据集大小（10个数据） 即把训练集随机按每10个分组，抽30次运行
    net.SDG(list(training_data), 30, 10, 3.0, test_data=list(test_data))