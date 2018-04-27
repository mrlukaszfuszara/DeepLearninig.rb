class DenseLayer
  attr_reader :batch_size, :output, :weights, :delta, :error

  def initialize(batch_size, activation, last_size)
    @mm = MatrixMath.new
    @a = Activations.new
    @c = Costs.new
    @g = Generators.new
    @activation = activation
    @last_size = last_size
    @batch_size = batch_size
  end

  def compile_data
    create_weights
  end

  def fit_forward(input = nil)
    @output = apply_activation(input)
    @output = calc_forward
  end

  def fit_backward(layer = nil, data_y = nil, output = nil, weights = nil, delta = nil)  	
    @layer = layer

    @weights_next = weights
    @delta_next = delta
    @output_last = output

    if layer
      @error = @c.quadratic_cost(data_y, @output)
      @delta = @mm.subt(@output, data_y)
    else
      mult = @mm.dot(@weights_next, @delta_next)
      deriv = apply_d(@output)
      @delta = @mm.mult(mult, deriv)
    end
    @delta_weights = @mm.div(@mm.mult(@delta, @output), @output.size)
  end

  def update_weights(alpha)
    @weights = @mm.subt(@weights, @mm.mult(@mm.mult([@output], @delta_weights)[0], alpha))
  end

  private

  def create_weights
    @weights = @g.random_matrix(@last_size, @batch_size, 0.0..1.0)
  end

  def calc_forward
    @mm.dot(@weights.transpose, @output)
  end

  def apply_activation(layer)
    tmp = 0
    if @activation == 'sigmoid'
      tmp = @a.sigmoid(layer)
    elsif @activation == 'tanh'
      tmp = @a.tanh(layer)
    elsif @activation == 'relu'
      tmp = @a.relu(layer)
    elsif @activation == 'leaky_relu'
      tmp = @a.leaky_relu(layer)
    end
    tmp
  end

  def apply_d(deriv)
    tmp = 0
    if @activation == 'sigmoid'
      tmp = @a.sigmoid_d(deriv)
    elsif @activation == 'tanh'
      tmp = @a.tanh_d(deriv)
    elsif @activation == 'relu'
      tmp = @a.relu_d(deriv)
    elsif @activation == 'leaky_relu'
      tmp = @a.leaky_relu_d(deriv)
    end
    tmp
  end
end
