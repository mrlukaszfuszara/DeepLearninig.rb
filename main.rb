require './lib/math'

require './lib/layers/dense'
require './lib/layers/rnn'

class Main
  attr_reader :output

  def initialize(batch_size, data_x, data_y, epochs, alpha)
    r = RNN.new
    r.add_rnn(batch_size, true)
    r.compile
    rnn = r.fit(data_x, data_y, alpha)
    d1 = Dense.new
    d1.add_dense(batch_size / 6, 'sigmoid')
    d1.add_dense(256, 'sigmoid')
    d1.add_dense(3, 'relu')
    d1.compile
    @output = d1.fit(rnn, data_y, epochs, alpha)
  end
end

f = Functions.new
data_x = f.random_matrix_small(100, 12)
data_y = [0.9, 0.4, 0.9]
epochs = 100
batch_size = data_x[0].size
alpha = 0.00001

p Main.new(batch_size, data_x, data_y, epochs, alpha).output
