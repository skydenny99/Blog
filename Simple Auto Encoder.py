import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import *
from keras.datasets import mnist
from keras.utils import to_categorical
from matplotlib import cm
import sys

def build_model():
    encoder_input = Input(shape=(784))
    dense1 = LeakyReLU()(Dense(units=256)(encoder_input))
    dense1 = LeakyReLU()(Dense(units=128)(dense1))
    dense2 = Dense(units=2)(dense1)
    decoder_input = Input(shape=(2))
    dense3 = LeakyReLU()(Dense(units=128)(decoder_input))
    dense3 = LeakyReLU()(Dense(units=256)(dense3))
    output = Dense(units=784, activation='sigmoid')(dense3)

    encoder = Model(encoder_input, dense2)
    decoder = Model(decoder_input, output)
    full_model = Model(encoder_input, decoder(dense2))
    return full_model, encoder, decoder

def preprocess():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = (x_train.astype('float32') / 255)
    x_test = (x_test.astype('float32') / 255)

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test))

def show_img(sample):
    sample = np.reshape(sample, newshape=(28,28))
    two_d = (sample * 255).astype('uint8')
    plt.imshow(two_d, interpolation='nearest')
    plt.show()

if __name__ == '__main__':
    x, y = preprocess()
    x = np.reshape(x, newshape=(x.shape[0], 784))
    full, enc, dec = build_model()
    training = False

    if training:
        full.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])
        full.fit(x, x, batch_size=32, epochs=10, validation_split=0.1)
        full.save_weights('2ae.h5')
    else:
        full.load_weights('2ae.h5')

    fig = plt.figure(figsize=(12, 10))
    lg = []
    for i in range(0, 10):
        number = i
        num_filter = np.where(np.argmax(y, axis=1) == number)
        tmp_x = x[num_filter]
        print(tmp_x.shape)
        result_x = np.transpose(enc.predict(tmp_x))
        lg.append(plt.scatter(result_x[0], result_x[1], s=1, alpha=0.8))
    lgnd = plt.legend(lg, range(0, 10), loc='upper left', fontsize='large')
    for handle in lgnd.legendHandles:
        handle.set_sizes([10.0])
    plt.show()

    a, b = map(float, sys.stdin.readline().strip().split())
    while not (a == 0 and b == 0):
        tmp = np.reshape(np.array([a, b]), newshape=(1, 2))
        show_img(dec.predict(tmp))
        a, b = map(float, sys.stdin.readline().strip().split())