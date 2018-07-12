import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
#from keras.utils.vis_utils import plot_model 


def build_multilayer_perceptron():
    model = Sequential()

    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model


def plot_history(history):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid(
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.show()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.show()


if __name__ == "__main__":
    batch_size = 128
    nb_classes = 10
    nb_epoch = 100

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = build_multilayer_perceptron()

    model.summary()
    #plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(patience=0, verbose=1)

    history = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        nb_epoch=nb_epoch,
                        verbose=1,
                        validation_split=0.1,
                        callbacks=[early_stopping])

    plot_history(history)

    loss, acc = model.evaluate(X_test, Y_test, verbose=0)

    print('Test loss:', loss)
    print('Test acc:', acc)