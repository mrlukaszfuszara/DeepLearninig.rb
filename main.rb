require './lib/util/splitter_tdt'
require './lib/util/splitter_mb'
require './lib/util/normalization'

require './lib/util/matrix_math'
require './lib/util/generators'
require './lib/util/activations'
require './lib/util/costs'

require './lib/nn/nn'

class Main
  def train(epochs, data_x, data_y, cost_function, optimizer, learning_rate, iterations, regularization_l2)
    if optimizer == 'gd'
      nn = NN.new(data_x[0].size)
      nn.add_nn(12, 'leaky_relu')
      nn.add_nn(24, 'leaky_relu', 0.4)
      nn.add_nn(24, 'leaky_relu', 0.4)
      nn.add_nn(24, 'leaky_relu', 0.4)
      nn.add_nn(1, 'leaky_relu')
      nn.compile(data_x.size)
      tmp = nn.fit(epochs, data_x, data_y, cost_function, optimizer, learning_rate, iterations, regularization_l2)
      nn.save_weights('./test')
      tmp
    elsif optimizer == 'mini-batch-gd'
      nn = NN.new(data_x[0][0].size)
      nn.add_nn(12, 'leaky_relu')
      nn.add_nn(24, 'leaky_relu', 0.4)
      nn.add_nn(24, 'leaky_relu', 0.4)
      nn.add_nn(24, 'leaky_relu', 0.4)
      nn.add_nn(1, 'leaky_relu')
      nn.compile(data_x[0].size)
      tmp = nn.fit(epochs, data_x, data_y, cost_function, optimizer, learning_rate, iterations, regularization_l2)
      nn.save_weights('./test')
      tmp
    end
  end

  def predict(data_x, data_y, cost_function, regularization_l2)
    nn = NN.new(data_x[0].size)
    nn.add_nn(12, 'leaky_relu')
    nn.add_nn(24, 'leaky_relu')
    nn.add_nn(24, 'leaky_relu')
    nn.add_nn(24, 'leaky_relu')
    nn.add_nn(1, 'leaky_relu')
    nn.load_weights('./test')
    nn.predict(data_x, data_y, cost_function, regularization_l2)
  end
end

@mm = MatrixMath.new
g = Generators.new
data_x = g.random_matrix(30, 3, 0.0..1.0)
data_y = g.random_vector(30, 0.0..1.0)

#data_x = [[0.1, 0.7, 0.1],[0.1, 0.2, 0.1],[0.1, 0.3, 0.1],[0.1, 0.6, 0.1],[0.1, 0.2, 0.1]]
#data_y = [0.1,0.1,0.1,0.1,0.5]

stdt = SpliterTDT.new(data_x, data_y)
train_set = stdt.train
dev_set = stdt.dev
test_set = stdt.dev

train_set_x = train_set[0]
train_set_y = train_set[1]

mini_batch_size = 15
smb_train = SplitterMB.new(mini_batch_size, train_set_x, train_set_y)
train_set_x = smb_train.data_x
train_set_y = smb_train.data_y

dev_set_x = dev_set[0]
dev_set_y = dev_set[1]
test_set_x = test_set[0]
test_set_y = test_set[1]

optimizer = 'mini-batch-gd'
cost_function = 'mse'
learning_rate = 0.1
epochs = 100
regularization_l2 = 0.1
iterations = 1

main = Main.new
p main.train(epochs, train_set_x, train_set_y, cost_function, optimizer, learning_rate, iterations, regularization_l2)

p main.predict(dev_set_x, dev_set_y, cost_function, regularization_l2)
