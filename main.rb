require './lib/Math/activations'
require './lib/Math/costs'
require './lib/Math/generators'
require './lib/Math/matrix_math'

require './lib/NeuralNetwork/neural_network'

class Main
  attr_reader :output

  def initialize(data_x, data_y, epochs, alpha)
    d1 = NeuralNetwork.new
    d1.add_dense(6, 'leaky_relu')
    d1.add_dense(24, 'leaky_relu')
    d1.add_dense(12, 'leaky_relu')
    d1.add_dense(3, 'leaky_relu')
    d1.compile
    @output = d1.fit(data_x, data_y, epochs, alpha)
  end
end

g = Generators.new
data_x = g.random_matrix(100, 6, 0.0..0.1).transpose
data_x = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]].transpose
data_y = [0.5, 0.1, 0.9]
epochs = 10000
alpha = 0.00000001

p Main.new(data_x, data_y, epochs, alpha).output
