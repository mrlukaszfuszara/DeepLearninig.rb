class NN
  attr_reader :error

  def initialize(data_x_size = nil, weights = nil)
    @mm = MatrixMath.new
    @g = Generators.new
    @a = Activations.new
    @c = Costs.new

    @array_of_layers = []
    @array_of_activations = []
    @array_of_dropouts = []
    @array_of_weights = []
    @array_of_bias = []

    add_nn(data_x_size, 'nil')

    @features = data_x_size
  end

  def add_nn(batch_size, activation, dropout = 1.0)
    @array_of_layers << Array.new(batch_size)
    @array_of_activations << activation
    @array_of_dropouts << dropout
  end

  def compile(samples)
    @samples = samples
    i = 1
    while i < @array_of_layers.size
      @array_of_weights << create_weights(i)
      i += 1
    end
    i = 0
    while i < @array_of_layers.size - 1
      @array_of_bias << create_bias(i + 1)
      i += 1
    end
  end

  def fit(train_data_x, train_data_y, cost_function, alpha, epochs, iterations, regularization_l2 = nil)
    @regularization_l2 = regularization_l2
    epochs.times do
      @array_of_z = []
      @array_of_a = []

      @array_of_delta_a = []
      @array_of_delta_z = []
      @array_of_delta_w = []
      @array_of_delta_b = []

      fit_forward(train_data_x, 0)
      i = 1
      while i < @array_of_layers.size - 1
        fit_forward(@array_of_z[i - 1], i)
        if i == @array_of_layers.size - 2
          puts 'Train Error: ' + apply_cost(cost_function, @array_of_a[i].flatten, train_data_y, i).to_s
        end
        i += 1
      end
      array_of_d = []
      i = 0
      while i < @array_of_a.size
        array_of_d[i] = []
        tmp = @g.random_matrix(@array_of_a[i].size, @array_of_a[i][0].size, 0.0..0.1)
        j = 0
        while j < tmp.size
          array_of_d[i][j] = []
          k = 0
          while k < tmp[j].size
            if tmp[j][k] < @array_of_dropouts[i]
              array_of_d[i][j][k]  = 1.0
            else
              array_of_d[i][j][k]  = 0.0
            end
            k += 1
          end
          j += 1
        end
        @array_of_a[i] = @mm.mult(@mm.mult(@array_of_a[i], array_of_d[i]), (1.0 / @array_of_dropouts[i])) #/
        i += 1
      end
      iterations.times do
        i = @array_of_layers.size - 1
        while i > 1
          fit_backward_step_one(i, train_data_y)
          i -= 1
        end
        i = @array_of_layers.size - 1
        while i > 1
          fit_backward_step_two(i - 1, alpha)
          i -= 1
        end
      end
    end
    p @array_of_a.last
  end

  def save_weights(path)
    serialized_array1 = Marshal.dump(@array_of_weights)
    File.open(path + '_w.msh', 'wb') { |f| f.write(serialized_array1) }
    serialized_array2 = Marshal.dump(@array_of_bias)
    File.open(path + '_b.msh', 'wb') { |f| f.write(serialized_array2) }
  end

  def load_weights(path)
    @array_of_weights = Marshal.load File.open(path + '_w.msh', 'rb')
    @array_of_bias = Marshal.load File.open(path + '_b.msh', 'rb')
  end

  def predict(dev_data_x, dev_data_y, cost_function, regularization_l2 = nil)
    @regularization_l2 = regularization_l2
    @array_of_z = []
    @array_of_a = []
    fit_forward(dev_data_x, 0)
    i = 1
    while i < @array_of_layers.size - 1
      fit_forward(@array_of_z[i - 1], i)
      if i == @array_of_layers.size - 2
        puts 'Prediction Error: ' + apply_cost(cost_function, @array_of_a[i].flatten, dev_data_y, i).to_s
      end
      i += 1
    end
    @array_of_a.last
  end

  private

  def create_weights(counter)
    @mm.mult(@g.random_matrix(@array_of_layers[counter].size, @array_of_layers[counter - 1].size, 0.0..0.01), Math.sqrt(2.0 / @features)) #/
  end

  def create_bias(counter)
    @g.zero_vector(@array_of_layers[counter].size)
  end

  def fit_forward(z, counter)
    @array_of_z[counter] = @mm.add_reversed(@mm.dot(@array_of_weights[counter], z), @array_of_bias[counter])
    @array_of_a[counter] = apply_a(@array_of_z[counter], counter + 1)
  end

  def fit_backward_step_one(counter, data_y)
    @array_of_delta_a[counter] = @mm.subt(@array_of_a[counter - 1], data_y) if counter == @array_of_layers.size - 1
    @array_of_delta_z[counter] = @mm.mult(@array_of_delta_a[counter], apply_d(@array_of_z[counter - 1], counter))
    @array_of_delta_a[counter - 1] = @mm.dot(@array_of_weights[counter - 1].transpose, @array_of_delta_z[counter])
    if !@regularization_l2.nil?
      tmp = @mm.mult(@mm.dot(@array_of_delta_z[counter], @array_of_a[counter - 2].transpose), (1.0 / @samples)) #/
      @array_of_delta_w[counter] = @mm.add(tmp, @mm.mult(@array_of_weights[counter - 1], (@regularization_l2 / @samples))) #/
    else
      @array_of_delta_w[counter] = @mm.mult(@mm.dot(@array_of_delta_z[counter], @array_of_a[counter - 2].transpose), (1.0 / @samples)) #/
    end
    @array_of_delta_b[counter] = @mm.mult(@mm.vertical_sum(@array_of_delta_z[counter]), (1.0 / @samples)) #/
  end

  def fit_backward_step_two(counter, alpha)
    @array_of_weights[counter] = @mm.subt(@array_of_weights[counter], @mm.mult(@array_of_delta_w[counter + 1], alpha))
    @array_of_bias[counter] = @mm.subt(@array_of_bias[counter], @mm.mult(@array_of_delta_b[counter + 1], alpha))
  end

  def apply_cost(cost_function, data_x, data_y, counter)
    tmp1 = 0
    if !@regularization_l2.nil?
      tmp2 = @mm.f_norm(@array_of_weights[counter])
      if cost_function == 'mse'
        tmp1 = @c.quadratic_cost_with_r(@array_of_a[counter].flatten, data_y, data_x.size, @regularization_l2, tmp2)
      elsif cost_function == 'cross_entropy'
        tmp1 = @c.cross_entropy_cost_with_r(@array_of_a[counter].flatten, data_y, data_x.size, @regularization_l2, tmp2)
      end
    else
      if cost_function == 'mse'
        tmp1 = @c.quadratic_cost(@array_of_a[counter].flatten, data_y, data_x.size)
      elsif cost_function == 'cross_entropy'
        tmp1 = @c.cross_entropy_cost(@array_of_a[counter].flatten, data_y, data_x.size)
      end
    end
    tmp1
  end

  def apply_a(z, counter)
    if @array_of_activations[counter] == 'nil'
      tmp = z
    elsif @array_of_activations[counter] == 'relu'
      tmp = @a.relu(z)
    elsif @array_of_activations[counter] == 'leaky_relu'
      tmp = @a.leaky_relu(z)
    elsif @array_of_activations[counter] == 'tanh'
      tmp = @a.tanh(z)
    elsif @array_of_activations[counter] == 'sigmoid'
      tmp = @a.sigmoid(z)
    end
    tmp
  end

  def apply_d(z, counter)
    if @array_of_activations[counter] == 'nil'
      tmp = z
    elsif @array_of_activations[counter] == 'relu'
      tmp = @a.relu_d(z)
    elsif @array_of_activations[counter] == 'leaky_relu'
      tmp = @a.leaky_relu_d(z)
    elsif @array_of_activations[counter] == 'tanh'
      tmp = @a.tanh_d(z)
    elsif @array_of_activations[counter] == 'sigmoid'
      tmp = @a.sigmoid_d(z)
    end
    tmp
  end
end
