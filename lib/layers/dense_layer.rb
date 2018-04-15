class DenseLayer
  attr_reader :batch_size, :output, :weights, :delta, :error

  def initialize(batch_size, activation, last_size)
    @f = Functions.new
    @activation = activation
    @last_size = last_size
    @batch_size = batch_size
  end

  def compile_data
    create_weights
  end

  def prepare(data_x = nil, data_y = nil)
    @data_x = data_x
    @data_y = data_y
  end

  def fit_forward(output = nil)
    @data_x = output unless output.nil?
    @output = apply_activation(calc_forward)
  end

  def fit_backward(layer = nil, output = nil, weights = nil, delta = nil)
    @layer = layer

    @weights_next = weights
    @delta_next = delta
    @output_last = output

    if layer == 1
      @delta = @f.subt(@output, @data_y)
    elsif layer.zero?
      deriv = apply_d(@output_last)
      mult = @f.mult(@weights, @output)
      @delta = @f.dot(mult.transpose, deriv)
    end

    if layer == 1
      @error = @f.mse_error(@output, @data_y)
    end
  end

  def update_weights(update)
    alpha = 0.01

    @weights = @f.subt(@weights, @f.mult(@f.mult(@output, update), alpha))
  end

  private

  def create_weights
    @weights = @f.random_matrix_full(@last_size, @batch_size)
    @weights
  end

  def calc_forward
    @f.dot(@weights.transpose, @data_x)
  end

  def apply_activation(layer)
    tmp = 0
    if @activation == 'sigmoid'
      tmp = @f.sigmoid(layer)
    elsif @activation == 'tanh'
      tmp = @f.tanh(layer)
    elsif @activation == 'relu'
      tmp = @f.relu(layer)
    end
    tmp
  end

  def apply_d(deriv)
    tmp = 0
    if @activation == 'sigmoid'
      tmp = @f.sigmoid_d(deriv)
    elsif @activation == 'tanh'
      tmp = @f.tanh_d(deriv)
    elsif @activation == 'relu'
      tmp = @f.relu_d(deriv)
    end
    tmp
  end
end