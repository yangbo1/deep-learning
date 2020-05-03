import random

import numpy as np


def sigmoid(z):
    # s型函数
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    # s的导数
    return sigmoid(z)*(1-sigmoid(z))

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
        if test_data:
            print("epoch : {0} / {1}".format(self.evaluate(test_data), n_test))

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
            # 反向传播，计算所有层的w和b的误差
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            # 全是0的wb各加上误差
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            # 调整w和b
            """
            w = w - η/m * ∑误差
            b = b - η/m * ∑误差
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
        :param x: 输入，第一层为输入x，后面为计算的a
        :param y: 对应正确输出结果
        :return:
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # a = S(z)
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            # 计算激活函数a=S(z)
            # z = (w·a+b)
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # C = 1/2 ∑ (y -a)² =》 C对a的导数为 (a-y)
        # △C(即δ) = (a-y)· a导
        # 即误差的变化量公式：C关于a的变化率
        # 通过结果误差传播给最后一层的a 即 w和b的误差
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

        # 求出输出层上的b和与前一层之间的w
        # △b = δ 因为C = 1/2 ∑ (y -a)²  a = (w·a+b)所以C对b的偏导数等于对a的偏导数 ∂C/∂b = ∂C/∂a * ∂a/∂b 又 ∂a/∂b = 1
        nabla_b[-1] = delta
        # △w = δ · a 因为∂C/∂w = ∂C/∂a * ∂a/∂w 又 ∂a/∂w = a
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # 一层层往前传
        for i in range(2, self.num_layers):
            z = zs[-i]
            sp = sigmoid_prime(z)
            # 根据上面的计算最后一层的误差来计算出前一层误差
            # δ = 后面w转置 · 误差 * a导
            delta = np.dot(self.weights[-i+1].transpose(), delta) * sp
            # 同理计算出前一层的b误差和w误差
            nabla_b[-i] = delta
            nabla_w[-i] = np.dot(delta, activations[-i-1].transpose())
        # 返回所有层的b和w的误差
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