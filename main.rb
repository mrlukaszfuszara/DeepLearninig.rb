require 'io/console'
require 'csv'
require 'matrix'

require './lib/util/splitter_mini_batch'
require './lib/neural_network/neural_network'
require './lib/neural_network/conv_network'
require './lib/neural_network/recurrent_network'

# require './lib/util/splitter_train_dev_test'
require './lib/util/normalization'

class Main
  def train_seqnet(seq_x, seq_y)
    seqnet = RecurrentNetwork.new
    seqnet.input
    seqnet.add_recnet(seq_x[0].size, seq_y[0].size)
    seqnet.compile(true)
    out_seq = seqnet.fit(seq)
    seqnet.save_weights('./weights/weights_resnet.msh')
    seqnet.save_architecture('./weights/arch_resnet.msh')
    out_seq
  end

  def train_conv(images_path, images)
    convnet = ConvNetwork.new
    convnet.input
    convnet.add_convnet('leaky_relu', 48, 8, 0, 4)
    convnet.add_maxpool(3, 0, 2)
    convnet.add_convnet('leaky_relu', 128, 4, 2, 1)
    convnet.add_maxpool(3, 0, 2)
    convnet.add_convnet('leaky_relu', 196, 4, 1, 1)
    convnet.add_convnet('leaky_relu', 196, 4, 1, 1)
    convnet.add_convnet('leaky_relu', 128, 4, 1, 1)
    convnet.add_maxpool(3, 0, 2)
    convnet.compile
    convnet.fit(images_path, images)
    img_x = convnet.return_flatten
    convnet.save_weights('./weights/weights_convnet.msh')
    convnet.save_architecture('./weights/arch_convnet.msh')
    img_x
  end

  def train_neuralnet(data_x, data_y, epochs, iterations, cost_function, optimizer, learning_rate, decay_rate, momentum)
    neuralnet = NeuralNetwork.new
    neuralnet.input(data_x[0][0].size)
    neuralnet.add_neuralnet(256, 'leaky_relu', 0.5)
    neuralnet.add_neuralnet(256, 'leaky_relu', 0.5)
    neuralnet.add_neuralnet(data_y[0][0].size, 'softmax')
    neuralnet.compile(optimizer, cost_function, learning_rate, decay_rate, momentum)
    tmp = neuralnet.fit(data_x, data_y, epochs, iterations)
    neuralnet.save_weights('./weights/weights_neuralnet.msh')
    neuralnet.save_architecture('./weights/arch_neuralnet.msh')
    tmp
  end

  def predict_conv
    convnet = ConvNetwork.new
    convnet.load_architecture('./weights/arch_convnet.msh.sha512', './weights/arch_convnet.msh')
    convnet.load_weights('./weights/weights_convnet.msh.sha512', './weights/weights_convnet.msh')
    convnet.predict(images_path, images)
    convnet.return_flatten
  end

  def predict_neuralnet(data_x, data_y)
    neuralnet = NeuralNetwork.new
    neuralnet.load_architecture('./weights/arch_neuralnet.msh.sha512', './weights/arch_neuralnet.msh')
    neuralnet.load_weights('./weights/weights_neuralnet.msh.sha512', './weights/weights_neuralnet.msh')
    neuralnet.predict(data_x, data_y)
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
img_y = g.one_hot_vector(g.tags_to_numbers(img_y))

batch_size = 12

smb = SplitterMiniBatch.new(img_x, img_y, batch_size)
img_x = smb.x
img_y = smb.y

output = Marshal.dump(img_x)
File.open('tmpx.msh', 'wb') { |f| f.write(output) }
output = Marshal.dump(img_y)
File.open('tmpy.msh', 'wb') { |f| f.write(output) }

img_x = Marshal.load File.open('tmpx.msh', 'rb')
img_y = Marshal.load File.open('tmpy.msh', 'rb')

n = Normalization.new(img_x)
n.z_score
n.min_max_scaler
img_x = n.matrix

network = Main.new
epochs = 10
iterations = 10
optimizer = 'Adam'
cost_function = 'crossentropy'
learning_rate = 0.001
decay_rate = 1
momentum = [0.9, 0.999, 10**-8]
network.train_neuralnet(img_x, img_y, epochs, iterations, cost_function, optimizer, learning_rate, decay_rate, momentum)

network = Main.new
network.predict_neuralnet(img_x, img_y)
