require './lib/neural_network/network'

require './lib/util/matrix_math'
require './lib/util/generators'
require './lib/util/activations'
require './lib/util/costs'

class RecurrentNetwork < Network
  def initialize
    @activation = Activations.new
    @cost = Costs.new

    @w_f_weights_array = []
    @w_i_weights_array = []
    @w_c_weights_array = []
    @w_o_weights_array = []
    @w_y_weights_array = []
  end

  def input; end

  def add_recnet(seq_x_size, seq_y_size)
    @seq_array << [seq_x_size, seq_y_size]
  end

  def compile(full_return)
    @full_return = full_return

    i = 0
    while i < @seq_array.size
      @w_f_weights_array << create_weights_c(i)
      @w_i_weights_array << create_weights_c(i)
      @w_c_weights_array << create_weights_c(i)
      @w_o_weights_array << create_weights_c(i)
      @w_y_weights_array << create_weights_y(i)
      i += 1
    end
  end

  def fit(train_seq)
    puts 'Lets start!'

    @c = Array.new(@seq_array.size, [])
    @o = Array.new(@seq_array.size, [])
    @y_hat = Array.new(@seq_array.size, [])

    i = 0
    while i < train_seq.size
      x = Matrix[*train_seq[i]]

      forward_propagation(x)

      i += 1
    end
  end

  def predict(sample_seq)
    sample_seq
  end

  private

  def create_weights_c(counter)
    Array.new(@seq_array[counter][0] + @seq_array[counter][1]) { Array.new(@seq_array[counter][0], 0.0) }
  end

  def create_weights_y(counter)
    Array.new(@seq_array[counter][0]) { Array.new(@seq_array[counter][1], 0.0) }
  end

  def forward_propagation(data_x)
    layer = 0
    while layer < @seq_array.size
      data_x = @h[layer].concatenate(data_x)

      hf = apply_activ(data_x * @w_f_weights_array[layer], 'sigmoid')
      hi = apply_activ(data_x * @w_i_weights_array[layer], 'sigmoid')
      ho = apply_activ(data_x * @w_o_weights_array[layer], 'sigmoid')
      hc = apply_activ(data_x * @w_c_weights_array[layer], 'tanh')

      @c[layer] = hf.hadamard_product(@c[layer]) + hi.hadamard_product(hc)
      @h[layer] = ho.hadamard_product(apply_activ(@c[layer], 'tanh'))
      @y_hat[layer] = @h[layer] * @w_y_weights_array[layer]

      @probability = apply_activ(@y_hat[layer], 'softmax')
      layer += 1
    end
  end

  def apply_activ(layer, activation)
    tmp = nil
    if activation == 'relu'
      tmp = @activation.relu(layer)
    elsif activation == 'leaky_relu'
      tmp = @activation.leaky_relu(layer)
    elsif activation == 'softmax'
      tmp = @activation.softmax(layer)
    elsif activation == 'tanh'
      tmp = @activation.tanh(layer)
    elsif activation == 'sigmoid'
      tmp = @activation.sigmoid(layer)
    end
    tmp
  end

  def apply_deriv(layer, activation)
    tmp = nil
    if activation == 'relu'
      tmp = @activation.relu_d(layer)
    elsif activation == 'leaky_relu'
      tmp = @activation.leaky_relu_d(layer)
    elsif activation == 'nil'
      tmp = layer
    end
    tmp
  end
end