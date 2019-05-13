from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


class Dataset:
    def __init__(self, batch_size=128):
        self.mnist = input_data.read_data_sets("./", one_hot=True)
        self.num_samples = self.mnist.train._num_examples
        self.batch_size = batch_size
        self.num_batches = int(self.num_samples / self.batch_size)
        print(self.num_samples)

    def get_test_data(self):
        return (self.mnist.test.images, self.mnist.test.labels)

    def get_test_data_batch(self):
        return self.mnist.test.next_batch(self.batch_size)

    def get_train_data(self):
        return self.mnist.train

    def get_validation_data(self):
        return (self.mnist.validation.images, self.mnist.validation.labels)

class DatasetExclude:
    def __init__(self, batch_size=128, exclude_class=[3]):
        self.mnist = input_data.read_data_sets("./", one_hot=True)
        self.num_samples = self.mnist.train._num_examples
        self.batch_size = batch_size
        self.num_batches = int(self.num_samples / self.batch_size)
        self.exclude_class=exclude_class
        self.test_data_ex = self.get_test_data_ex()
        print(self.num_samples)

    def get_test_data(self):
        return (self.mnist.test.images, self.mnist.test.labels)
    
    def get_test_data_ex(self):
        test_data = self.mnist.test
        arrX = np.empty(test_data.images.shape)
        arrY = np.empty(test_data.labels.shape)
        for ex_class in self.exclude_class:
            Xdata_ex = np.array([x for (x,y) in zip(test_data.images, test_data.labels) if y[ex_class]==1])
            ydata_ex = np.array([y for y in test_data.labels if y[ex_class]==1])
            arrX = np.concatenate((arrX, Xdata_ex))
            arrY = np.concatenate((arrY, ydata_ex))
        return (arrX, arrY)

    def get_test_data_batch(self):
        return self.mnist.test.next_batch(self.batch_size)
    
    def get_test_data_batch_ex(self, batch_size=None):
        num = self.batch_size if batch_size is None else batch_size
        data = self.test_data_ex[0]
        labels = self.test_data_ex[1]
        idx = np.arange(0 , len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = data[idx]
        labels_shuffle = labels[idx]

        return data_shuffle, labels_shuffle

    def get_train_data(self):
        return TrainData(self.mnist, self.num_samples, self.batch_size, self.exclude_class)

    def get_validation_data(self):
        return (self.mnist.validation.images, self.mnist.validation.labels)
    
class TrainData:
    def __init__(self, mnist, num_samples, batch_size, ex_class=[3]):
        self.mnist = mnist
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_batches = int(self.num_samples / self.batch_size)
        self.ex_class=ex_class
        self.ex_data = self.exclude_class()
        
    def exclude_class(self):
        train_data = self.mnist.train
        arrX = train_data.images
        arrY = train_data.labels
        for ex_class in self.exclude_class:
            arrX = np.array([x for (x,y) in zip(arrX, arrY) if y[ex_class]==0])
            arrY = np.array([y for y in arrY if y[ex_class]==0])
        return (arrX, arrY)
        
    def next_batch(self, batch_size=None):
        num = self.batch_size if batch_size is None else batch_size
        data = self.ex_data[0]
        labels = self.ex_data[1]
        idx = np.arange(0 , len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = data[idx]
        labels_shuffle = labels[idx]

        return data_shuffle, labels_shuffle

class DataSetTinyImageNet():
    