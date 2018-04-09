class Dense
  def initialize(activation, size, last_size)
    @f = Functions.new
    @activation = activation
    @size = size
    @last_size = last_size
  end

  def compile_data(data_x)
    if @last_size == 0
      @input_size = data_x.size
      @input = data_x
    else
      @input_size = @size
    end
    
    create_weights
  end

  def fit_forward(data_x)
    if @last_size == 0
      @input_size = data_x.size
      @input = data_x
    else
      @input_size = @size
      @input = data_x
    end

    dot_forward_data
    apply_activation

    @output
  end

  def fit_backward(weights, delta, output, data_y = nil, last_layer = nil)
    alpha = 0.01

    @weights_next = weights
    @delta_next = delta
    @output_last = output

    @data_y = data_y
    @last_layer = last_layer

    if last_layer
      tmp_0 = @f.sub(@output, @data_y)
      # tmp_0 return: Vector
      tmp_1 = @f.mult(@weights, @output)
      # tmp_1 return: Matrix
      tmp_2 = apply_d(tmp_1)
      # tmp_2 return: Matrix
      tmp_3 = @f.dot(tmp_2, tmp_0)
      # tmp_3 return: Vector
      @delta = tmp_3
    else
      tmp_0 = @f.mult(@weights_next.transpose, @delta_next).transpose
      # tmp_0 return: Matrix
      tmp_1 = @f.mult(@weights, @output_last)
      # tmp_1 return: Matrix
      tmp_2 = apply_d(tmp_1)
      # tmp_2 return: Matrix
      tmp_3 = @f.dot(tmp_2, tmp_0)
      # tmp_3 return: Vector
      @delta = tmp_3
    end

    if last_layer
      tmp_0 = @f.mult(@delta, @f.slice_vector(@output).transpose)
    else
      tmp_0 = @f.mult(@delta, @f.slice_vector(@output_last).transpose)
    end

    tmp_1 = @f.mult(tmp_0, alpha)

    @weights -= tmp_1

    [@weights, @delta, @output]
  end



  private

  def create_weights
    if @last_size == 0
      @weights = @f.random_matrix_full(@input_size, @size)
    else
      @weights = @f.random_matrix_full(@last_size, @input_size)
    end
  end

  def dot_forward_data
    @output = @f.dot(@weights.transpose, @input)
  end

  def apply_activation
    tmp = 0
    if @activation == 'sigmoid'
      tmp = @f.sigmoid(@output)
    elsif @activation == 'tanh'
      tmp = @f.tanh(@output)
    elsif @activation == 'relu'
      tmp = @f.relu(@output)
    end
    @output = tmp
  end

  def apply_d(d)
    tmp = 0
    if @activation == 'sigmoid'
      tmp = @f.sigmoid_d(d)
    elsif @activation == 'tanh'
      tmp = @f.tanh_d(d)
    elsif @activation == 'relu'
      tmp = @f.relu_d(d)
    end
    output = tmp
    output
  end
end