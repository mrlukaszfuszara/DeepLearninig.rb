require './lib/math'

require './lib/layers/dense'

class Main
  attr_reader :output

  def initialize(batch_size, data_x, data_y, epochs, alpha)
    d1 = Dense.new
    d1.add_dense(batch_size, 'sigmoid')
    d1.add_dense(256, 'sigmoid')
    d1.add_dense(3, 'relu')
    d1.compile
    @output = d1.fit(data_x, data_y, epochs, alpha)
  end
end

f = Functions.new
data_x = f.random_matrix_small(100, 50)
data_y = [0.9, 0.4, 0.9]
epochs = 100
batch_size = 10
alpha = 0.000001

p Main.new(batch_size, data_x, data_y, epochs, alpha).output
