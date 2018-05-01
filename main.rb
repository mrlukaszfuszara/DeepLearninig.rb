require './lib/util/splitter_tdt'
require './lib/util/splitter_mb'
require './lib/util/normalization'

require './lib/util/matrix_math'
require './lib/util/generators'
require './lib/util/activations'
require './lib/util/costs'

require './lib/nn/nn'

class Main
  def train(data_x, data_y, cost_function, optimizer, learning_rate, iterations, regularization_l2, batch_size = nil)
    if optimizer == 'gd'
      nn = NN.new(data_x[0].size, data_x.size)
      nn.add_nn(12, 'leaky_relu')
      nn.add_nn(24, 'leaky_relu', 0.4)
      nn.add_nn(24, 'leaky_relu', 0.4)
      nn.add_nn(24, 'leaky_relu', 0.4)
      nn.add_nn(1, 'leaky_relu')
      nn.compile
      tmp = nn.fit(data_x, data_y, cost_function, optimizer, learning_rate, iterations, regularization_l2)
      nn.save_weights('./test')
      tmp
    elsif optimizer == 'mini-batch-gd'
      nn = NN.new(data_x[0].size, batch_size)
      nn.add_nn(12, 'leaky_relu')
      nn.add_nn(24, 'leaky_relu', 0.8)
      nn.add_nn(1, 'leaky_relu')
      nn.compile
      tmp = nn.fit(data_x, data_y, cost_function, optimizer, learning_rate, iterations, regularization_l2, batch_size)
      nn.save_weights('./weights.msh')
      nn.save_architecture('./arch.msh')
      tmp
    end
  end

  def predict(data_x, data_y, cost_function, regularization_l2)
    nn = NN.new(data_x[0].size)
    nn.load_architecture('./arch.msh')
    nn.load_weights('./weights.msh')
    nn.predict(data_x, data_y, cost_function, regularization_l2)
  end
end

@mm = MatrixMath.new
g = Generators.new
data_x = g.random_matrix(900, 3, 0.0..1.0)
data_y = g.random_vector(900, 0.0..1.0)

#data_x = [[0.1, 0.7, 0.1],[0.1, 0.2, 0.1],[0.1, 0.3, 0.1],[0.1, 0.6, 0.1],[0.1, 0.2, 0.1],[0.1, 0.7, 0.1],[0.1, 0.2, 0.1],[0.1, 0.3, 0.1],[0.1, 0.6, 0.1],[0.1, 0.2, 0.1],[0.1, 0.7, 0.1],[0.1, 0.2, 0.1],[0.1, 0.3, 0.1],[0.1, 0.6, 0.1],[0.1, 0.2, 0.1],[0.1, 0.7, 0.1],[0.1, 0.2, 0.1],[0.1, 0.3, 0.1],[0.1, 0.6, 0.1],[0.1, 0.2, 0.1],[0.1, 0.7, 0.1],[0.1, 0.2, 0.1],[0.1, 0.3, 0.1],[0.1, 0.6, 0.1],[0.1, 0.2, 0.1],[0.1, 0.7, 0.1],[0.1, 0.2, 0.1],[0.1, 0.3, 0.1],[0.1, 0.6, 0.1],[0.1, 0.2, 0.1]]
#data_y = [0.1,0.1,0.1,0.1,0.5,0.1,0.1,0.1,0.1,0.5,0.1,0.1,0.1,0.1,0.5,0.1,0.1,0.1,0.1,0.5,0.1,0.1,0.1,0.1,0.5,0.1,0.1,0.1,0.1,0.5]

stdt = SpliterTDT.new(data_x, data_y)
train_set = stdt.train
dev_set = stdt.dev
test_set = stdt.dev

train_set_x = train_set[0]
train_set_y = train_set[1]

batch_size = 2

dev_set_x = dev_set[0]
dev_set_y = dev_set[1]
test_set_x = test_set[0]
test_set_y = test_set[1]

optimizer = 'mini-batch-gd'
cost_function = 'mse'
learning_rate = 0.001
regularization_l2 = 0.01
iterations = 10

main = Main.new
p main.train(train_set_x, train_set_y, cost_function, optimizer, learning_rate, iterations, regularization_l2, batch_size)

p main.predict(dev_set_x, dev_set_y, cost_function, regularization_l2)
