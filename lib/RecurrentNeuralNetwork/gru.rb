require './lib/layers/gru_layer'

class GRU
  def add_rnn(batch_size, full_weights = false)
    @nn = GRULayer.new(batch_size)
    @full_weights = full_weights
  end

  def compile
    @nn.compile_data
  end

  def fit(data_x)
    if @full_weights
      tmp = []
      i = 0
      while i < data_x.size
        tmp[i] = @nn.fit_forward(data_x[i])
        i += 1
      end
    else
      i = 0
      while i < data_x.size
        tmp = @nn.fit_forward(data_x[i])
        i += 1
      end
    end
    tmp
  end
end