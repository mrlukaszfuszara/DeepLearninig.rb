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
    cn.input('leaky_relu', 16, 3, 1, 2)
    cn.add_convnet('leaky_relu', 16, 3, 1, 2)
    cn.add_maxpool(2, 0, 2)
    cn.add_convnet('leaky_relu', 16, 3, 1, 2)
    cn.compile
    cn.fit(images_path)
    img_x = cn.return_flatten
    cn.save_weights('./weights/weights_cn.msh')
    cn.save_architecture('./weights/arch_cn.msh')
    img_x
  end
  def train_nn(data_x, data_y, batch_size, epochs, dev_data, cost_function, optimizer, learning_rate, decay_rate, iterations, momentum, regularization_l2)
    nn = NeuralNetwork.new
    nn.input(data_x[0].size, 'leaky_relu')
    nn.add_neuralnet(32, 'leaky_relu', 0.7)
    nn.add_resnet(8, 4, 8, 'leaky_relu')
    nn.add_neuralnet(33, 'softmax')
    nn.compile(optimizer, cost_function, learning_rate, decay_rate, iterations, momentum, regularization_l2)
    tmp = nn.fit(data_x, data_y, batch_size, epochs, dev_data)
    nn.save_weights('./weights/weights_nn.msh')
    nn.save_architecture('./weights/arch_nn.msh')
    tmp
  end

  def predict_conv(images_path)

  end

  def predict_nn(data_x, data_y, batch_size)
    nn = NeuralNetwork.new
    nn.load_architecture('./weights/arch_nn.msh')
    nn.load_weights('./weights/weights_nn.msh')
    nn.predict(data_x, data_y, batch_size)
  end
end

img = '.\dataset\images'
network = Main.new

#img_x = network.train_conv(img)
#output = Marshal.dump(img_x)
#File.open('ConvNet.msh', 'wb') { |f| f.write(output) }

img_x = Marshal.load File.open('ConvNet.msh', 'rb')

files = Marshal.load File.open('./data/sequence_of_img.msh', 'rb')

labels = []
CSV.foreach('./data/validation-annotations-bbox.csv', { :col_sep => ',' }) do |row|
  labels << row
end

img_y = []

i = 0
while i < files.size
  j = 0
  while j < labels.size
    if files[i] == (labels[j][0] + '.png')
      img_y[i] = labels[j][2]
    end
    j += 1
  end
  i += 1
end

@g = Generators.new
img_y = @g.tags_to_numbers(img_y)
img_y = @g.one_hot_vector(img_y)

network = Main.new
epochs = 12
optimizer = 'Adam'
cost_function = 'crossentropy'
learning_rate = 0.5
regularization_l2 = nil
iterations = 40
decay_rate = 1
momentum = [0.9, 0.999, 10**-8]
batch_size = 10
network.train_nn(img_x, img_y, batch_size, epochs, nil, cost_function, optimizer, learning_rate, decay_rate, iterations, momentum, regularization_l2)

network = Main.new
network.predict_nn(img_x, img_y, 10)
