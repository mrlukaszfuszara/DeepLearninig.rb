require './lib/layers/rnn_layer'

class RNN
  def add_rnn(batch_size)
    @nn = RNNLayer.new(batch_size)
  end

  def compile
    @nn.compile_data
  end

  def fit(data_x, data_y, alpha)
    i = 0
    while i < data_x.size
      tmp = @nn.fit_forward(data_x[i])
      i += 1
    end
    tmp
  end
end