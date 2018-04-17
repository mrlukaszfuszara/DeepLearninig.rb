class DenseLayer
  attr_reader :batch_size, :output_forward, :output, :weights, :delta, :error

  def initialize(batch_size, activation, last_size)
    @f = Functions.new
    @activation = activation
    @last_size = last_size
    @batch_size = batch_size
  end

  def compile_data
    create_weights
  end

  def fit_forward(output = nil)
    @output_forward = apply_activation(output)
    @output_forward = calc_forward
  end

  def fit_backward(layer = nil, data_y = nil, output = nil, weights = nil, delta = nil)
    @layer = layer

    @weights_next = weights
    @delta_next = delta
    @output_last = output

    if layer
      @error = @f.mse_error(@output_forward, data_y)
      @delta = @f.subt(@output_forward, data_y)
    else
      deriv = apply_d(@output_last)
      mult = @f.mult(@weights, @output_forward)
      @delta = @f.dot(mult.transpose, deriv)
    end
  end

  def update_weights(update, alpha)
    @weights = @f.subt(@weights, @f.mult(@f.mult(@output_forward, update), alpha))

    @output = @output_forward
  end

  private

  def create_weights
    @weights = @f.random_matrix_full(@last_size, @batch_size)
    @weights
  end

  def calc_forward
    @f.dot(@weights.transpose, @output_forward)
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