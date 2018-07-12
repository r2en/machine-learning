import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import toimage
from keras.datasets import cifar10

def main():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    nclassed = 10
    pos = 1

    for target_class in range(nclassed):
        target_idx = []
        for i in range(len(y_train)):
            if y_train[i][0] == target_class:
                target_idx.append(i)

        np.random.shuffle(target_idx)
        for idx in target_idx[:10]:
            img = toimage(x_train[idx])
            plt.subplot(10, 10, pos)
            plt.imshow(img)
            plt.axis('off')
            pos += 1
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()