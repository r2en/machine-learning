import os
import sys
import numpy as np
from scipy.misc import toimage
import matplotlib.pyplot as plt

from keras.datasets import cifar10
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

def save_history(history, result_file):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(result_file, 'w') as fp:
        fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
        for i in range(nb_epoch):
            fp.write("%d\t%f\t%f\t%f\t%f\n" % (i, loss[i], acc[i], val_loss[i], val_acc[i]))


def build_cnn(x_train, nb_classes):
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model

def main():
    '''
    if len(sys.argv) != 4:
        print('usage: python cifar10-cnn.py [nb_epoch][use_data_augmentation(True or False)][result_dir]')
        exit(1)

    nb_epoch = int(sys.argv[1])
    data_augmentation = True if sys.argv[2] == 'True' else False
    result_dir = sys.argv[3]
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    '''

    nb_epoch = 30
    data_augmentation = False
    image_dir = 'cifar10'
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)

    batch_size = 128
    nb_classes = 10

    img_rows, img_cols = 32, 32
    img_channels = 3

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /=255.0

    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    #model = build_cnn(x_train, nb_classes)
    #model = load_model(image_dir)

    model_file = os.path.join(image_dir, 'model.json')
    weight_file = os.path.join(image_dir, 'model.h5')

    with open(model_file, 'r') as fp:
        model = model_from_json(fp.read())
    model.load_weights(weight_file)

    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        nb_epoch=nb_epoch,
        verbose=1,
        validation_data=(x_test, y_test),
        shuffle=True
    )


    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model_json = model.to_json()
    with open(os.path.join(image_dir, 'model.json'), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(image_dir, 'model.h5'))
    save_history(history, os.path.join(image_dir, 'history.txt'))

    loss, acc = model.evaluate(x_test, y_test, verbose=0)

    print('test loss: ', loss)
    print('test acc: ', acc)

    #import pdb; pdb.set_trace()



if __name__ == '__main__':
    main()
