require 'csv'

require './lib/util/splitter_tdt'
require './lib/util/splitter_mb'
require './lib/util/normalization'

require './lib/util/matrix_math'
require './lib/util/generators'
require './lib/util/activations'
require './lib/util/costs'

require './lib/nn/nn'

class Main
  def train(epochs, data_x, data_y, cost_function, optimizer, learning_rate, iterations, decay_rate, regularization_l2, batch_size = nil)
    nn = NN.new(data_x[0].size, batch_size)
    nn.add_nn(8, 'leaky_relu', 0.9)
    nn.add_nn(8, 'leaky_relu', 0.9)
    nn.add_nn(1, 'leaky_relu')
    nn.compile
    tmp = nn.fit(epochs, data_x, data_y, cost_function, optimizer, learning_rate, iterations, decay_rate, regularization_l2, batch_size)
    nn.save_weights('./weights.msh')
    nn.save_architecture('./arch.msh')
    tmp
  end

  def predict(data_x, data_y, cost_function, regularization_l2, batch_size)
    nn = NN.new(data_x[0].size)
    nn.load_architecture('./arch.msh')
    nn.load_weights('./weights.msh')
    nn.predict(data_x, data_y, cost_function, regularization_l2, batch_size)
  end
end

g = Generators.new
#data_x = g.random_matrix(1_000, 30, 0.0..1.0)
#data_y = g.random_vector(1_000, 0.0..1.0)

#data_x = [[0.1, 0.7, 0.1],[0.1, 0.2, 0.1],[0.1, 0.3, 0.1],[0.1, 0.6, 0.1],[0.1, 0.2, 0.1],[0.1, 0.7, 0.1],[0.1, 0.2, 0.1],[0.1, 0.3, 0.1],[0.1, 0.6, 0.1],[0.1, 0.2, 0.1],[0.1, 0.7, 0.1],[0.1, 0.2, 0.1],[0.1, 0.3, 0.1],[0.1, 0.6, 0.1],[0.1, 0.2, 0.1],[0.1, 0.7, 0.1],[0.1, 0.2, 0.1],[0.1, 0.3, 0.1],[0.1, 0.6, 0.1],[0.1, 0.2, 0.1],[0.1, 0.7, 0.1],[0.1, 0.2, 0.1],[0.1, 0.3, 0.1],[0.1, 0.6, 0.1],[0.1, 0.2, 0.1],[0.1, 0.7, 0.1],[0.1, 0.2, 0.1],[0.1, 0.3, 0.1],[0.1, 0.6, 0.1],[0.1, 0.2, 0.1]]
#data_y = [0.1,0.1,0.1,0.1,0.5,0.1,0.1,0.1,0.1,0.5,0.1,0.1,0.1,0.1,0.5,0.1,0.1,0.1,0.1,0.5,0.1,0.1,0.1,0.1,0.5,0.1,0.1,0.1,0.1,0.5]

tmp = []
CSV.foreach('./dataset/winequality-white.csv', { :col_sep => ';' }) do |row|
  tmp << row
end

data_x = []
data_y = []
i = 0
while i < tmp.size
  data_x[i] = []
  j = 0
  while j < tmp[0].size
    if j == tmp[0].size - 1
      data_y << tmp[i][j].to_f
    else
      data_x[i][j] = tmp[i][j].to_f
    end
    j += 1
  end
  i += 1
end

stdt = SpliterTDT.new(data_x, data_y)
train_set = stdt.train
dev_set = stdt.dev
test_set = stdt.dev

train_set_x = train_set[0]
train_set_y = train_set[1]

batch_size = 32

dev_set_x = dev_set[0]
dev_set_y = dev_set[1]
test_set_x = test_set[0]
test_set_y = test_set[1]

n = Normalization.new
train_set_x = n.subt_mean(train_set_x)
mean = n.mean

n = Normalization.new
dev_set_x = n.subt_mean(dev_set_x, mean)

epochs = 4
optimizer = 'Adam'
cost_function = 'mse'
learning_rate = 0.00001
regularization_l2 = 0.1
iterations = 30
decay_rate = 1

test_x = [[6.3,0.48,0.04,1.1,0.046,30,99,0.9928,3.24,0.36,9.6]]
test_y = [8]

main = Main.new
main.train(epochs, train_set_x, train_set_y, cost_function, optimizer, learning_rate, iterations, decay_rate, regularization_l2, batch_size,)

#tmp = main.predict(test_x, test_y, cost_function, regularization_l2)
#p tmp

p main.predict(dev_set_x, dev_set_y, cost_function, regularization_l2, batch_size)
