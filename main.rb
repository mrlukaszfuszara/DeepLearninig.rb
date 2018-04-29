require './lib/Math/matrix_math'
require './lib/Math/generators'
require './lib/Math/activations'
require './lib/Math/costs'

require './lib/NN/nn'

class Main
  attr_reader :output

  def initialize(data_x, data_y, cost_function, alpha, epochs)
    nn = NN.new(data_x[0].size)
    nn.add_nn(12, 'leaky_relu')
    nn.add_nn(1, 'leaky_relu')
    nn.compile(data_x.size)
    @output = nn.fit(data_x, data_y, cost_function, alpha, epochs)
  end
end

data_x = [[0.3, 0.2, 0.5, 0.1], [0.3, 0.2, 0.5, 0.2], [0.3, 0.2, 0.5, 0.3], [0.3, 0.2, 0.5, 0.6], [0.3, 0.2, 0.5, 0.8], [0.3, 0.2, 0.5, 0.1]]
@mm = MatrixMath.new
#g = Generators.new
#data_x = g.random_matrix(100, 3, 0.0..0.1)
data_y = [0.3, 0.9, 0.5, 0.6, 0.1, 0.7]
cost_function = 'mse'
alpha = 10**-2
epochs = 1_000
p Main.new(data_x, data_y, cost_function, alpha, epochs).output
