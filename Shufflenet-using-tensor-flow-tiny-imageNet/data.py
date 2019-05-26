import data_processing
import tensorflow as tf
import numpy as np

class load_data():
    def __init__(self, images, labels, _num_examples):
        self.images = images
        self.labels = labels
        self._num_examples = _num_examples

class Dataset:
    def __init__(self, batch_size=128):
        classes, x_train, y_train, x_val, y_val = data_processing.load_tiny_imagenet(num_classes=200, resize='true')
        #self.train.images, self.train.labels
        self.train = load_data(x_train, y_train, x_train.shape[0])
        self.val = load_data(x_val, y_val, x_val.shape[0])

        self.num_samples = self.train._num_examples
        self.batch_size = batch_size
        self.num_batches = int(self.num_samples / self.batch_size)
        print(f'sample size: {self.num_samples}')

#     def get_test_data(self):
#         return (self.mnist.test.images, self.mnist.test.labels)

#     def get_test_data_batch(self):
#         return self.mnist.test.next_batch(self.batch_size)

    def get_train_data(self):  
        return self.train

    def get_validation_data(self):
        return (self.validation.images, self.validation.labels)

    def next_batch(self, batch_size=None):
        num = self.batch_size if batch_size is None else batch_size
        data = self.train.images
        labels = self.train.labels
        idx = np.arange(0 , len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = data[idx]
        labels_shuffle = labels[idx]
        return data_shuffle, labels_shuffle