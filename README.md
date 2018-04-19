# RuNNet
Ruby Neural Networks 

Class Main

Method initialize
* There is definied architecture of neural network.
* Available layers: Dense, RNN
* Available activation functions: sigmoid, tanh, relu
* Available loss functions: MSE
* Available optimizers of backpropagation: SGD
* Dense: add_dense. This method get two parameters: number of X matrix samples in one step and activation function
* RNN: add_rnn. This method get two parameters: number of vectors in X matrix and return of RNN (if true all weights, if false only last weight)

Class Functions

Mathematics and Utils library

Other
* data_x - input matrix
* data_y - backpropagation matrix with optimal parameters for neural network
* epochs - forward and backward step for all samples
* batch_size - initial size of input matrix
* alpha - learning rate for optimizer
