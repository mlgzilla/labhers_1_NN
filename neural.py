import numpy
import numpy as np


class Perceptron:
    def __init__(self, inputSize, hiddenSizes, outputSize):

        self.Win = np.zeros((1 + inputSize, hiddenSizes))
        self.Win[0, :] = (np.random.randint(0, 3, size=(hiddenSizes)))
        self.Win[1:, :] = (np.random.randint(-1, 2, size=(inputSize, hiddenSizes)))

        self.Wout = np.random.randint(0, 2, size=(1 + hiddenSizes, outputSize)).astype(np.float64)
        # self.Wout = np.random.randint(0, 3, size = (1+hiddenSizes,outputSize))

    def predict(self, Xp):
        hidden_predict = np.where((np.dot(Xp, self.Win[1:, :]) + self.Win[0, :]) >= 0.0, 1, -1).astype(np.float64)
        out = np.where((np.dot(hidden_predict, self.Wout[1:, :]) + self.Wout[0, :]) >= 0.0, 1, -1).astype(np.float64)
        return out, hidden_predict

    def train(self, X, y, n_iter=5, eta=0.01):
        prev = []
        num_mist = 1
        while all(not np.array_equal(self.Wout, x) for x in prev) and num_mist != 0:
            prev.append(self.Wout.copy())
            print(self.Wout.reshape(1, -1))
            num_mist = 0
            for xi, target, j in zip(X, y, range(X.shape[0])):
                pr, hidden = self.predict(xi)
                if target - pr != 0:
                    num_mist += 1
                self.Wout[1:] += ((eta * (target - pr)) * hidden).reshape(-1, 1)
                self.Wout[0] += eta * (target - pr)
        if num_mist == 0:
            print("Было достигнуто условие сходимости перцептрона")
        else:
            print("Было достигнуто условие зацикливания перцептрона")
        return self



