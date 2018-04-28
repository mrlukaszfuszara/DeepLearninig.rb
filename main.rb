require './lib/Math/matrix_math'
require './lib/Math/generators'
require './lib/Math/activations'
require './lib/Math/costs'

require './lib/NN/nn'

class Main
  attr_reader :output

  def initialize(data_x, data_y, cost_function, alpha, epochs)
    nn = NN.new(data_x.size)
    nn.add_nn(12, 'leaky_relu')
    nn.add_nn(24, 'leaky_relu')
    nn.add_nn(6, 'leaky_relu')
    nn.add_nn(1, 'leaky_relu')
    nn.compile(data_x[0].size)
    @output = nn.fit(data_x, data_y, cost_function, alpha, epochs)
  end
end

data_x = [[0.3, 0.2, 0.5], [0.3, 0.2, 0.5], [0.3, 0.2, 0.5], [0.3, 0.2, 0.5], [0.3, 0.2, 0.5], [0.3, 0.2, 0.5]]
g = Generators.new
data_x = g.random_matrix(100, 3, 0.0..0.1)
data_y = [10, 0, 30]
cost_function = 'mse'
alpha = 0.00001
epochs = 20000
p Main.new(data_x, data_y, cost_function, alpha, epochs).output
