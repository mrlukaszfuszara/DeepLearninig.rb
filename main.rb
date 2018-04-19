require './lib/math'

require './lib/layers/dense'
require './lib/layers/rnn'

class Main
  attr_reader :output

  def initialize(batch_size, data_x, data_y, epochs, alpha)
# For Dense
=begin
    d1 = Dense.new
    d1.add_dense(batch_size, 'sigmoid')
    d1.add_dense(256, 'sigmoid')
    d1.add_dense(3, 'relu')
    d1.compile
    @output = d1.fit(data_x, data_y, epochs, alpha)
=end
    r = RNN.new
    r.add_rnn(batch_size)
    r.compile
    @output = r.fit(data_x, data_y, alpha)
  end
end

=begin
f = Functions.new
data_x = f.random_matrix_small(100, 50)
data_y = [0.9, 0.4, 0.9]
epochs = 100
batch_size = 10
alpha = 0.000001

p Main.new(batch_size, data_x, data_y, epochs, alpha).output
=end

f = Functions.new
data_x = f.random_matrix_small(100, 12)
data_y = [0.9, 0.4, 0.9]
epochs = 100
batch_size = data_x[0].size
alpha = 0.000001

p Main.new(batch_size, data_x, data_y, epochs, alpha).output
