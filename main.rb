require './lib/math'

require './lib/layers/dense'

class Main
  attr_reader :output

  def initialize(batch_size, data_x, data_y, epochs)
    d1 = Dense.new
    d1.add_dense(batch_size, 'sigmoid')
    d1.add_dense(16, 'tanh')
    d1.add_dense(64, 'sigmoid')
    d1.add_dense(3, 'relu')
    d1.compile
    @output = d1.fit(data_x, data_y, 100)
  end
end

f = Functions.new
data_x = f.random_matrix_small(100, 50)
data_y = [0.9, 0.4, 0.9]
epochs = 1000
batch_size = 2

p Main.new(batch_size, data_x, data_y, epochs).output
