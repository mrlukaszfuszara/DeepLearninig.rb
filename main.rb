require 'io/console'
require 'csv'
require 'matrix'
# require 'pp'

require './lib/util/splitter_mini_batch'
require './lib/neural_network/neural_network'
require './lib/neural_network/conv_network'

# require './lib/util/splitter_train_dev_test'
# require './lib/util/normalization'

class Main
  def train_conv(images_path, images)
    cn = ConvNetwork.new
    cn.input('leaky_relu', 3, 5, 1, 2)
    cn.add_convnet('leaky_relu', 4, 3, 1, 2)
    cn.add_maxpool(3, 0, 1)
    cn.add_convnet('leaky_relu', 18, 3, 2, 1)
    cn.compile
    cn.fit(images_path, images)
    img_x = cn.return_flatten
    cn.save_weights('./weights/weights_cn.msh')
    cn.save_architecture('./weights/arch_cn.msh')
    img_x
  end
  def train_nn(data_x, data_y, batch_size, epochs, dev_data, cost_function, optimizer, learning_rate, decay_rate, momentum)
    nn = NeuralNetwork.new
    nn.input(data_x[0][0].size)
    nn.add_neuralnet(16, 'leaky_relu', 0.5)
    nn.add_neuralnet(16, 'leaky_relu', 0.5)
    nn.add_neuralnet(data_y[0][0].size, 'softmax')
    nn.compile(optimizer, cost_function, learning_rate, decay_rate, momentum)
    tmp = nn.fit(data_x, data_y, epochs)
    nn.save_weights('./weights/weights_nn.msh')
    nn.save_architecture('./weights/arch_nn.msh')
    tmp
  end

  def predict_conv(images_path, images)
    cn = ConvNetwork.new
    cn.load_architecture('./weights/arch_cn.msh')
    cn.load_weights('./weights/weights_cn.msh')
  end

  def predict_nn(data_x, data_y)
    nn = NeuralNetwork.new
    nn.load_architecture('./weights/arch_nn.msh')
    nn.load_weights('./weights/weights_nn.msh')
    nn.predict(data_x, data_y)
  end
end


g = Generators.new

g.generate_images_path('./dataset/images', './data/images.msh')
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

batch_size = 8

smb = SplitterMiniBatch.new(img_x, img_y, batch_size)
img_x = smb.x
img_y = smb.y

network = Main.new
epochs = 3
optimizer = 'Adam'
cost_function = 'crossentropy'
learning_rate = 0.0001
decay_rate = 1
momentum = [0.9, 0.999, 10**-8]
network.train_nn(img_x, img_y, batch_size, epochs, nil, cost_function, optimizer, learning_rate, decay_rate, momentum)

network = Main.new
network.predict_nn(img_x, img_y)
