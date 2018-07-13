import os
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import tensorflow as tf
from keras import backend as K
from tensorflow.python import debug as tf_debug

#K.set_session(tf_debug.LocalCLIDebugWrapperSession(tf.Session()))

from sklearn.preprocessing import MinMaxScaler

def build_cnn(input_shape, nb_filters, filter_size, pool_size, nb_classes):
    model = Sequential()

    model.add(Convolution2D(nb_filters,
                            filter_size[0], filter_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters, filter_size[0], filter_size[1]))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model


def visualize_filter(model, nb_filters, outdir, step):
    #import pdb; pdb.set_trace()
    W = model.layers[0].get_weights()[0]

    if K.image_dim_ordering() == 'tf':
        W = W.transpose(3, 2, 0, 1)

    nb_filter, nb_channel, nb_row, nb_col = W.shape

    plt.figure()
    for i in range(nb_filters):
        im = W[i, 0]

        scaler = MinMaxScaler(feature_range=(0, 255))
        im = scaler.fit_transform(im)

        plt.subplot(4, 8, i + 1)
        plt.axis('off')
        plt.imshow(im, cmap="gray")
    
    if step == 'before':
        plt.savefig(os.path.join(outdir, 'visual0.png'))
    elif step == 'after':
        plt.savefig(os.path.join(outdir, 'visual1.png'))
    #plt.show()
    plt.close()

def plot_history(history, outdir):
    plt.figure()
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(outdir, 'acc.png'))
    #plt.show()
    

def nomalization(X_train, Y_train, X_test, Y_test, img_rows, img_cols, nb_classes):
    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)

    return {
        'X_train':X_train, 'Y_train':Y_train,
        'X_test':X_test, 'Y_test':Y_test,
        'input_shape':input_shape
    }

def save_model(model, image_dir, history):
    model_json = model.to_json()
    with open(os.path.join(image_dir, 'model.json'), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(image_dir, 'model.h5'))
    save_history(history, os.path.join(image_dir, 'history.txt'))

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

def load_cnn(image_dir):
    model_file = os.path.join(image_dir, 'model.json')
    weight_file = os.path.join(image_dir, 'model.h5')

    with open(model_file, 'r') as fp:
        model = model_from_json(fp.read())
    model.load_weights(weight_file)
    return model

def main():
    batch_size = 128
    nb_classes = 10
    nb_epoch = 30

    img_rows, img_cols = 28, 28
    nb_filters = 32
    filter_size = (5, 5)
    pool_size = (2, 2)

    image_dir = 'learned_model'
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    
    data = nomalization(X_train, Y_train, X_test, Y_test, img_rows, img_cols, nb_classes)

    '''
    if os.path.exists(os.path.join(image_dir, 'model.h5')):
        model = load_cnn(image_dir)
    else:
        model = build_cnn(data['input_shape'], nb_filters, filter_size, pool_size, nb_classes)
    '''
    model = build_cnn(data['input_shape'], nb_filters, filter_size, pool_size, nb_classes)

    model.summary()
    #plot(model, show_shapes=True, to_file='result_mnist/model.png')

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    visualize_filter(model, nb_filters, image_dir, 'before')
    
    early_stopping = EarlyStopping()

    history = model.fit(data['X_train'], data['Y_train'],
                        batch_size=batch_size,
                        nb_epoch=nb_epoch,
                        verbose=1,
                        validation_split=0.1,
                        callbacks=[early_stopping])

    visualize_filter(model, nb_filters, image_dir, 'after')

    save_model(model, image_dir, history)
    plot_history(history, image_dir)

    loss, acc = model.evaluate(data['X_test'], data['Y_test'], verbose=0)

    print('Test loss:', loss)
    print('Test acc:', acc)

    #import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()