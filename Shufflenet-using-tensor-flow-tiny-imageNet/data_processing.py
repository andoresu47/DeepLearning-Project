import os
import pickle
import scipy.misc
import numpy as np
import imageio
import skimage
import tensorflow as tf
from PIL import Image


def load_tiny_imagenet(resize='False', num_classes=200, dtype=np.float32):
    """
    Load TinyImageNet. Each of TinyImageNet-100-A, TinyImageNet-100-B, and
    TinyImageNet-200 have the same directory structure, so this can be used
    to load any of them.
    Inputs:
    - path: String giving path to the directory to load.
    - dtype: numpy datatype used to load the data.
    Returns: A tuple of
    - class_names: A list where class_names[i] is a list of strings giving the
    WordNet names for class i in the loaded dataset.
    - X_train: (N_tr, 3, 64, 64) array of training images
    - y_train: (N_tr,) array of training labels
    - X_val: (N_val, 3, 64, 64) array of validation images
    - y_val: (N_val,) array of validation labels
    - X_test: (N_test, 3, 64, 64) array of testing images.
    - y_test: (N_test,) array of test labels; if test labels are not available
    (such as in student code) then y_test will be None.
    """
    
    path = 'tiny-imagenet-200'
    # First load wnids
    wnids_file = os.path.join('tiny-imagenet-200', 'wnids.txt')
    with open(wnids_file, 'r') as f:
        wnids = [x.strip() for x in f]

    # Map wnids to integer labels
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

    # Use words.txt to get names for each class
    words_file = os.path.join('tiny-imagenet-200', 'words.txt')
   
    with open(words_file, 'r') as f:
        wnid_to_words = dict(line.split('\t') for line in f)
    
    for wnid, words in wnid_to_words.items():
        wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
    class_names = [wnid_to_words[wnid] for wnid in wnids]

    # Next load training data.
    X_train = []
    y_train = []
    for i, wnid in enumerate(wnids):
        if (i + 1) % 20 == 0:
            print('loading data for classes %d / %d' % (i + 1, len(wnids)))
            # To figure out the filenames we need to open the boxes file
        boxes_file = os.path.join(path, 'train', wnid, '%s_boxes.txt' % wnid)
        
        with open(boxes_file, 'r') as f:
            filenames = [x.split('\t')[0] for x in f]
        num_images = len(filenames)

        if resize.lower() == 'true':
            X_train_block = np.zeros((num_images, 3, 32, 32), dtype=dtype)
        else:
            X_train_block = np.zeros((num_images, 3, 64, 64), dtype=dtype)

        y_train_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int64)
        
        for j, img_file in enumerate(filenames):
            img_file = os.path.join(path, 'train', wnid, 'images', img_file)
            img = imageio.imread(img_file)
            
            if resize.lower() == 'true':

                img = skimage.transform.resize(img, (32, 32, 3))
            
            if img.ndim == 2:
            ## grayscale file
                if resize.lower() == 'true':
                    img.shape = (32, 32, 1)
                else:
                    img.shape = (64, 64, 1)
            X_train_block[j] = img.transpose(2, 0, 1)

        X_train.append(X_train_block)
        y_train.append(y_train_block)
      # We need to concatenate all training data
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

      # Next load validation data
    with open(os.path.join(path, 'val', 'val_annotations.txt'), 'r') as f:
        img_files = []
        val_wnids = []
        
        for line in f:
          # Select only validation images in chosen wnids set
          if line.split()[1] in wnids:
            img_file, wnid = line.split('\t')[:2]
            img_files.append(img_file)
            val_wnids.append(wnid)
        num_val = len(img_files)
        
        y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])

        if resize.lower() == 'true':
            X_val = np.zeros((num_val, 3, 32, 32), dtype=dtype)
        else:
            X_val = np.zeros((num_val, 3, 64, 64), dtype=dtype)

        for i, img_file in enumerate(img_files):
            img_file = os.path.join(path, 'val', 'images', img_file)
            img = imageio.imread(img_file)
            if resize.lower() == 'true':
                #img = scipy.misc.imresize(img, (32, 32, 3))
                img = skimage.transform.resize(img, (32, 32, 3))
            if img.ndim == 2:
                if resize.lower() == 'true':
                    img.shape = (32, 32, 1)
                else:
                    img.shape = (64, 64, 1)

            X_val[i] = img.transpose(2, 0, 1)
    y_train_ = tf.one_hot(y_train, len(y_train))
    y_val_ = tf.one_hot(y_val, len(y_val))

#     x_train_ = np.einsum('iljk->ijkl', X_train)
#     x_val_ = np.einsum('iljk->ijkl', X_val)
      # Omit x_test and y_test because they're unlabeled
      #return class_names, X_train, y_train, X_val, y_val, X_test, y_test
    return class_names, X_train, y_train_, X_val, y_val_

# #one hot label encoding
# train_y_onehot = tf.one_hot(y_train, len(y_train))
# val_y_onehot = tf.one_hot(y_val, len(y_val))
# #reshape data (N,32,32,3)
# x_train_ = np.einsum('iljk->ijkl', x_train)
# x_val_ = np.einsum('iljk->ijkl', x_val)