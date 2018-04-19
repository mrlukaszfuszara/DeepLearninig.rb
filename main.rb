require './lib/math'

require './lib/layers/dense'
require './lib/layers/rnn'
require './lib/layers/lstm'
require './lib/layers/gru'


class Main
  attr_reader :output

  def initialize(batch_size, data_x, data_y, epochs, alpha)
    r = GRU.new
    r.add_rnn(batch_size, true)
    r.compile
    rnn = r.fit(data_x)
    d1 = Dense.new
    d1.add_dense(batch_size / 3, 'sigmoid')
    d1.add_dense(24, 'sigmoid')
    d1.add_dense(12, 'sigmoid')
    d1.add_dense(3, 'relu')
    d1.compile
    @output = d1.fit(rnn, data_y, epochs, alpha)
  end
end

f = Functions.new
#data_x = [[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]]
data_x = f.random_matrix_small(100, 6)
data_y = [0.1, 0.9, 0.1]
epochs = 300
batch_size = data_x[0].size
alpha = 0.0001

p Main.new(batch_size, data_x, data_y, epochs, alpha).output
