require 'io/console'
require 'csv'

require 'chunky_png'

require './lib/util/image_loader'

require './lib/util/splitter_train_dev_test'
require './lib/util/splitter_mini_batch'
require './lib/util/normalization'
require './lib/util/vectorize_array'

require './lib/util/matrix_math'
require './lib/util/conv_math'

require './lib/util/generators'
require './lib/util/activations'
require './lib/util/costs'

require './lib/neural_network/neural_network'
require './lib/neural_network/conv_network'

class Main
  def initialize
    @g = Generators.new
  end

  def train_conv(images_path)
    cn = ConvNetwork.new
    cn.input('leaky_relu', 5, 3, 1, 1)
    cn.add_convnet('leaky_relu', 10, 3, 1, 2)
    cn.add_maxpool(3, 1, 2)
    cn.compile
    cn.fit(images_path)
    cn.return_flatten
    @img_x = cn.output
  end
  def train_nn(batch_size, epochs, dev_data, cost_function, optimizer, learning_rate, decay_rate, iterations, momentum, regularization_l2)
    data_y = @g.one_vector(@img_x[0].size)

    nn = NeuralNetwork.new
    nn.input(@img_x[0].size, 'leaky_relu')
    nn.add_neuralnet(32, 'leaky_relu', 0.7)
    nn.add_resnet(8, 4, 8, 'leaky_relu')
    nn.add_neuralnet(1, 'leaky_relu')
    nn.compile(optimizer, cost_function, learning_rate, decay_rate, iterations, momentum, regularization_l2)
    tmp = nn.fit(@img_x, data_y, batch_size, epochs, dev_data)
    nn.save_weights('./weights.msh')
    nn.save_architecture('./arch.msh')
    tmp
  end
end

img = 'C:\Users\Lukasz\Documents\Projekty\RuNNet\dataset\images'
network = Main.new
network.train_conv(img)

epochs = 5
optimizer = 'Adam'
cost_function = 'mse'
learning_rate = 0.5
regularization_l2 = nil
iterations = 60
decay_rate = 1
momentum = [0.9, 0.999, 10**-8]
batch_size = 3
network.train_nn(batch_size, epochs, nil, cost_function, optimizer, learning_rate, decay_rate, iterations, momentum, regularization_l2)

=begin
class Main
  def train(data_x, data_y, batch_size, epochs, dev_data, cost_function, optimizer, learning_rate, decay_rate, iterations, momentum, regularization_l2)
    nn = NeuralNetwork.new
    nn.input(data_x[0].size, 'leaky_relu')
    nn.add_neuralnet(32, 'leaky_relu', 0.7)
    nn.add_resnet(8, 4, 8, 'leaky_relu')
    nn.add_neuralnet(1, 'leaky_relu')
    nn.compile(optimizer, cost_function, learning_rate, decay_rate, iterations, momentum, regularization_l2)
    tmp = nn.fit(data_x, data_y, batch_size, epochs, dev_data)
    nn.save_weights('./weights.msh')
    nn.save_architecture('./arch.msh')
    tmp
  end

  def predict(data_x, data_y, batch_size)
    nn = NeuralNetwork.new
    nn.load_architecture('./arch.msh')
    nn.load_weights('./weights.msh')
    nn.predict(data_x, data_y, batch_size)
  end
end

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

#gen = Generators.new
#data_y = gen.one_hot_vector(data_y)

stdt = SpliterTrainDevTest.new(data_x, data_y)
train_set = stdt.train_s
dev_set = stdt.dev_s
test_set = stdt.test_s

train_set_x = train_set[0]
dev_set_x = dev_set[0]
test_set_x = test_set[0]

train_set_y = train_set[1]
dev_set_y = dev_set[1]
test_set_y = test_set[1]

batch_size = 64

n = Normalization.new
n.calculate(train_set_x)

train_set_x = n.z_score(train_set_x)
train_set_x = n.min_max_scaler(train_set_x)

dev_set_x = n.z_score(dev_set_x)
dev_set_x = n.min_max_scaler(dev_set_x)

test_set_x = n.min_max_scaler(test_set_x)

epochs = 5
optimizer = 'Adam'
cost_function = 'mse'
learning_rate = 0.5
regularization_l2 = nil
iterations = 60
decay_rate = 1
momentum = [0.9, 0.999, 10**-8]

main = Main.new
main.train(train_set_x, train_set_y, batch_size, epochs, [dev_set_x, dev_set_y], cost_function, optimizer, learning_rate, decay_rate, iterations, momentum, regularization_l2)

main.predict(test_set_x, test_set_y, batch_size)
=end
