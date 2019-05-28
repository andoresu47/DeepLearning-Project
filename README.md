# Knowledge Distillation

In multiple-class classification models, usually the objective function is the average log-probability. One feature of this type of learning is that the model assigns probabilities to every class, whose values can vary greatly even if they do not correspond to the correct label. This information can actually be useful to discern how the net generalizes. Since bigger and complicated models can be difficult to deploy, but usually perform well on complex tasks (such as image classification), it is this internal representation that we are interested in transmitting from a teacher model to a student model. 

# Project

This project consists of various experiments exploring the concept of **Knowledge Distillation**, introduced in [this paper](https://arxiv.org/abs/1503.02531) by Hinton et al., by using the teacher's class probabilities as "soft" inputs for training the student model. For this, we use a modified version of the regular *softmax* function by introducing a temperature factor which softens the teacher's probabilities. This approach was tested on a series of smaller models: a simple two-layered feed-forward network, a three-layered ConvNet and a ConvNet ensemble, for a variety of teachers: a self-trained 3 layer CNN, pre-trained networks based on the ResNet, VGG and ShuffleNet architectures. The experiments were performed on the the MNIST, Cifar100 and Tiny-ImageNet data sets. 

For the ShuffleNet experiments, the teacher model was taken from [here](https://github.com/TropComplique/ShuffleNet-tensorflow).

# Experimental Results 

Every folder in the repository leads to a set of experiments on specific teacher-student architectures and datasets. These are mainly condensed in jupyter notebooks, which make use of the scripts under the same directory. The table below shows sample accuracy results of using a small CNN as student model, both as standalone and after distilling knowledge from VGG16 and ResNet164 pre-trained nets, on the MNIST dataset. 

|     Model     | Test Accuracy |
| ------------- | ------------- |
| 3-layer CNN   | 97.00%  |
| VGG16         | 99.68%  |
| ResNet164     | 99.70%  |
| 3-Layer CNN + VGG16  | 97.65%  |
| 3-Layer CNN + ResNet164  | 98.16%  |
