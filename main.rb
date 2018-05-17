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

  def train_conv(images_path, images)
    cn = ConvNetwork.new
    cn.add_convnet('leaky_relu', 6, 3, 0, 4)
    cn.add_maxpool(3, 0, 2)
    cn.add_convnet('leaky_relu', 8, 3, 2, 1)
    cn.compile
    cn.fit(images_path, images)
    img_x = cn.return_flatten
    cn.save_weights('./weights/weights_cn.msh')
    cn.save_architecture('./weights/arch_cn.msh')
    img_x
  end
  def train_nn(data_x, data_y, batch_size, epochs, dev_data, cost_function, optimizer, learning_rate, decay_rate, iterations, momentum, regularization_l2)
    nn = NeuralNetwork.new
    nn.input(data_x[0][0].size, 'leaky_relu')
    nn.add_neuralnet(32, 'leaky_relu')
    nn.add_neuralnet(data_y[0][0].size, 'softmax')
    nn.compile(optimizer, cost_function, learning_rate, decay_rate, iterations, momentum, regularization_l2)
    tmp = nn.fit(data_x, data_y, batch_size, epochs, dev_data)
    nn.save_weights('./weights/weights_nn.msh')
    nn.save_architecture('./weights/arch_nn.msh')
    tmp
  end

  def predict_conv(images_path, images)

  end

  def predict_nn(data_x, data_y, batch_size)
    nn = NeuralNetwork.new
    nn.load_architecture('./weights/arch_nn.msh')
    nn.load_weights('./weights/weights_nn.msh')
    nn.predict(data_x, data_y, batch_size)
  end
end

g = Generators.new

img_x_array = g.generate_images_path('./dataset/images', './data/images.msh')
files = Marshal.load File.open('./data/images.msh', 'rb')

network = Main.new

img_x = network.train_conv('./dataset/images/', files)

output = Marshal.dump(img_x)
File.open('ConvNet.msh', 'wb') { |f| f.write(output) }

img_x = Marshal.load File.open('ConvNet.msh', 'rb')

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

g = Generators.new
img_y = g.tags_to_numbers(img_y)
img_y = g.one_hot_vector(img_y)

tmp = VectorizeArray.new
img_y = tmp.all(img_y)

batch_size = 2

smb = SplitterMiniBatch.new(batch_size, img_x, img_y)
img_x = smb.data_x
img_y = smb.data_y

n = Normalization.new
i = 0
while i < img_x.size
  img_x[i] = n.min_max_scaler(img_x[i])
  i += 1
end

network = Main.new
epochs = 10
optimizer = 'Adam'
cost_function = 'crossentropy'
learning_rate = 0.0001
regularization_l2 = nil
iterations = 20
decay_rate = 1
momentum = [0.9, 0.999, 10**-8]
network.train_nn(img_x, img_y, batch_size, epochs, nil, cost_function, optimizer, learning_rate, decay_rate, iterations, momentum, regularization_l2)

network = Main.new
network.predict_nn(img_x, img_y, batch_size)
