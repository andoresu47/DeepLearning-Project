from __future__ import print_function

from builtins import range
from six.moves import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from imageio import imread
import platform
import struct


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000,
                     subtract_mean=True):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. 
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }



def get_MNIST_data(num_training=59000, num_validation=1000, num_test=1000):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """
    path = 'datasets/MNIST'
    fname_img_train = os.path.join(path, 'train-images-idx3-ubyte')
    fname_lbl_train = os.path.join(path, 'train-labels-idx1-ubyte')
    fname_img_test = os.path.join(path, 't10k-images-idx3-ubyte')
    fname_lbl_test = os.path.join(path, 't10k-labels-idx1-ubyte')

    # Load everything in some numpy arrays
    with open(fname_lbl_train, 'rb') as flbl_train:
        _, _ = struct.unpack(">II", flbl_train.read(8))
        y_train = np.fromfile(flbl_train, dtype=np.int8)

    with open(fname_img_train, 'rb') as fimg_train:
        _, _, rows_train, cols_train = struct.unpack(">IIII", fimg_train.read(16))
        X_train = np.fromfile(fimg_train, dtype=np.uint8).reshape(len(y_train), rows_train, cols_train)
        
    with open(fname_lbl_test, 'rb') as flbl_test:
        _, _ = struct.unpack(">II", flbl_test.read(8))
        y_test = np.fromfile(flbl_test, dtype=np.int8)

    with open(fname_img_test, 'rb') as fimg_test:
        _, _, rows_test, cols_test = struct.unpack(">IIII", fimg_test.read(16))
        X_test = np.fromfile(fimg_test, dtype=np.uint8).reshape(len(y_test), rows_test, cols_test)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]
    
    # normalize data by centering and rescale to (0,1) interval
    _, H, W = X_train.shape
    X_train = np.reshape(X_train, (-1, H * W))
    mean_train = np.expand_dims(np.mean(X_train, 0), 0)
    std_train = np.expand_dims(np.std(X_train, 0), 0)
    
    X_train = (X_train - mean_train)/(std_train+1e-7)
    X_train = X_train.reshape(-1,H,W)
    
    X_val = np.reshape(X_val, (-1, H * W))
    X_val = (X_val - mean_train)/(std_train+1e-7)
    X_val = X_val.reshape(-1,H,W)
    
    X_test = np.reshape(X_test, (-1, H * W))
    X_test = (X_test - mean_train)/(std_train+1e-7)
    X_test = X_test.reshape(-1,H,W)
    
    X_train = X_train[:,np.newaxis,:,:]
    X_val = X_val[:,np.newaxis,:,:]
    X_test = X_test[:,np.newaxis,:,:]
    
    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }



def load_models(models_dir):
    """
    Load saved models from disk. This will attempt to unpickle all files in a
    directory; any files that give errors on unpickling (such as README.txt)
    will be skipped.

    Inputs:
    - models_dir: String giving the path to a directory containing model files.
      Each model file is a pickled dictionary with a 'model' field.

    Returns:
    A dictionary mapping model file names to models.
    """
    models = {}
    for model_file in os.listdir(models_dir):
        with open(os.path.join(models_dir, model_file), 'rb') as f:
            try:
                models[model_file] = load_pickle(f)['model']
            except pickle.UnpicklingError:
                continue
    return models


def show_MNIST(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    plt.show()
