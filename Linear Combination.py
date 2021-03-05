import tensorflow as tf
import numpy as np
from keras.layers import *
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

def build_model():
    input = Input(shape=(None, 784))
    dense = Dense(units=10, activation='softmax')(input)
    activate = Activation('softmax')(dense)
    model = Model(input, activate)
    non_active_model = Model(input, dense)
    model.summary()
    return model, non_active_model

def preprocess():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train = np.reshape(x_train, newshape=(x_train.shape[0], 784))
    x_test = np.reshape(x_test, newshape=(x_test.shape[0], 784))

    print(x_train.shape)
    print(x_test.shape)

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def show_img(sample):
    two_d = (np.reshape(sample, (28, 28)) * 255).astype('uint8')
    plt.imshow(two_d, interpolation='nearest')
    return plt

def show_graph(samples):
    for idx, val in enumerate(samples):
        plt.subplot(2, 5, idx+1)
        x = np.arange(10)
        max_val = max(samples[val])
        samples[val] = list(map(lambda x: x/max_val, samples[val]))
        plt.bar(x, samples[val])
        plt.xticks(x, x)
        plt.ylim(0, 1.05)
        plt.title(str(idx))
    # plt.subplot_tool()
    plt.show()

def calculate_dist_avg(crit, others):
    avg = np.average(np.sqrt(np.sum(np.power(others - crit, 2), axis=1)))
    return avg

def calculate_cos_avg(crit, others):
    result = []
    for i in others:
        result.append(np.dot(crit, i) / (np.linalg.norm(crit) * np.linalg.norm(i)))
    avg = np.average(result)
    return avg


def create_plot_data(x, y, similarity = 'dist'):
    sample = {}
    for i in range(10):
        sample[i] = []

    for i in range(10):
        crit = x[np.where(np.argmax(y, axis=1) == i)][0]
        # show_img(crit).show()
        for j in range(10):
            test_filter = np.where(np.argmax(y, axis=1) == j)
            test = x[test_filter]
            if similarity == 'dist':
                i_avg = calculate_dist_avg(crit, test)
            elif similarity == 'cos':
                i_avg = calculate_cos_avg(crit, test)
            sample[i].append(i_avg)
    return sample

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = preprocess()

    show_graph(create_plot_data(x_test, y_test))
    show_graph(create_plot_data(x_test, y_test, 'cos'))
    model, non_active_model = build_model()


    show_graph(create_plot_data(non_active_model.predict_on_batch(x_test), y_test))
    show_graph(create_plot_data(non_active_model.predict_on_batch(x_test), y_test, 'cos'))

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32, epochs=16, validation_split=0.1)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('loss: ', score[0])
    print('accuracy: ',score[1])

    show_graph(create_plot_data(non_active_model.predict_on_batch(x_test), y_test))
    show_graph(create_plot_data(non_active_model.predict_on_batch(x_test), y_test, 'cos'))
