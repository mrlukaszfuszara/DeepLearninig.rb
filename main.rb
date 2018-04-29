require './lib/util/splitter'

require './lib/util/matrix_math'
require './lib/util/generators'
require './lib/util/activations'
require './lib/util/costs'

require './lib/nn/nn'

class Main
  attr_reader :output

  def initialize(data_x, data_y, cost_function, learning_rate, epochs, regularization_l2)
    nn = NN.new(data_x[0].size)
    nn.add_nn(6, 'leaky_relu')
    nn.add_nn(12, 'leaky_relu')
    nn.add_nn(24, 'leaky_relu')
    nn.add_nn(12, 'leaky_relu')
    nn.add_nn(1, 'leaky_relu')
    nn.compile(data_x.size)
    @output = nn.fit(data_x, data_y, cost_function, learning_rate, epochs, regularization_l2)
  end
end

@mm = MatrixMath.new
g = Generators.new
data_x = g.random_matrix(50, 4, 0.0..0.1)
data_y = g.random_vector(50, 0.0..1.0)

s = Spliter.new(data_x, data_y)
train_set = s.train
dev_set = s.dev
test_set = s.dev

train_set_x = train_set[0]
train_set_y = train_set[1]
dev_set_x = dev_set[0]
dev_set_y = dev_set[1]
test_set_x = test_set[0]
test_set_y = test_set[1]

cost_function = 'mse'
learning_rate = 0.1
epochs = 100
regularization_l2 = 0.1

p Main.new(train_set_x, train_set_y, cost_function, learning_rate, epochs, regularization_l2).output
