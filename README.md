# DeepLearning.rb

In 'Main' class is defined architecture of neural network.

'train' method is used to create and fit neural network

'predict' method is used to predict

In 'Generators' class are defined methods to generate weights and bias, and one hot-vector generator.

'one_hot_vector' method is used to generate one-hot vector for Y data

In 'SplitterTDT' class are defined methods to split data sets.

In 'Normalization' class in defined z-score function.

Main train get parameters:
* X
* Y
* Batch Size
* Epochs
* Cost Function (mse {Y.size = 1}, log_loss {Y.size > 1})
* Optimizer (Gradient Descent {BGD}, Gradient Descent with Momentum {BGDwM}, RMSprop {RMSprop}, Adam {Adam})
* Learning Rate of optimizer
* Iterations of optimizer
* Decay Rate of Learning Rate
* Momentum of optimizer
* Regularization L2

Main predict get parameters:
* X
* Y
* Batch Size (the same as in train)
* Index of parameter to check accuracy

Architecture of neural network

To add layer use 'add_nn' method, first parameter is size of hidden units, second parameter is activation function (sigmoid, tanh, relu, leaky_relu for Y.size = 1 or softmax layer for Y.size > 1 on last)
