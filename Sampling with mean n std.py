import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.utils import to_categorical

def preprocess():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test))

def show_img(sample):
    two_d = sample.astype('uint8')
    plt.imshow(two_d, interpolation='nearest')
    plt.show()

if __name__ == '__main__':
    x, y = preprocess()
    for i in range(0, 10):
        number = i
        num_filter = np.where(np.argmax(y, axis=1) == number)
        tmp_x = x[num_filter]

        count = tmp_x.shape[0]
        result = tmp_x.transpose(1,2,0)
        result_mean = np.mean(result, axis=2)
        result_std = np.std(result, axis=2)

        for i in range(5):
            img = np.zeros(shape=(result.shape[0], result.shape[1]))
            for j in range(result.shape[0]):
                for k in range(result.shape[1]):
                    img[j][k] = np.clip(np.round(np.random.normal(result_mean[j][k], result_std[j][k])), 0, 255)
            show_img(img)
