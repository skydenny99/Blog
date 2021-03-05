import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
from keras.models import Model
from keras.layers import *
from keras.datasets import mnist
from keras.utils import to_categorical
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def preprocess():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = ((x_train.astype('float32') / 255) > 0.5).astype(int)
    x_test = ((x_test.astype('float32') / 255) > 0.5).astype(int)

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test))

def show_graph(data):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X = np.arange(0, data.shape[0])
    Y = np.arange(0, data.shape[1])
    X, Y = np.meshgrid(X, Y)
    Z = data

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)

    fig.colorbar(surf)
    plt.show()

def show_img(sample):
    two_d = (sample * 255).astype('uint8')
    plt.imshow(two_d, interpolation='nearest')
    plt.show()

if __name__ == '__main__':
    x, y = preprocess()
    for i in range(0, 10):
        number = i
        num_filter = np.where(np.argmax(y, axis=1) == number)
        tmp_x = x[num_filter]

        count = tmp_x.shape[0]
        result = np.sum(tmp_x.transpose(1,2,0), axis=2)
        # show_graph(result)
        result = result / count
        tmp = 1 - result
        for i in range(5):
            img = np.zeros(shape=(result.shape[0], result.shape[1]))
            for j in range(result.shape[0]):
                for k in range(result.shape[1]):
                    img[j][k] = np.argmax(np.random.multinomial(5, [tmp[j][k], result[j][k]]))
            show_img(img)
