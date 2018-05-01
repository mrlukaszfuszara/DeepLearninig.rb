class NN
  attr_reader :error

  def initialize(data_x_vertical_size, data_x_horizontal_size = nil)
    @mm = MatrixMath.new
    @g = Generators.new
    @a = Activations.new
    @c = Costs.new

    @array_of_layers = []
    @array_of_activations = []
    @array_of_dropouts = []
    @array_of_weights = []
    @array_of_bias = []

    @features = data_x_vertical_size
    @samples = data_x_horizontal_size

    add_nn(@features, 'nil')
  end

  def add_nn(batch_size, activation, dropout = 1.0)
    @array_of_layers << batch_size
    @array_of_activations << activation
    @array_of_dropouts << dropout
  end

  def compile
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

  def fit(train_data_x, train_data_y, cost_function, optimizer, alpha, iterations, regularization_l2 = nil, batch_size = nil, momentum = nil)
    @regularization_l2 = regularization_l2
    @cost_function = cost_function

    if optimizer == 'gd'
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
    elsif optimizer == 'mini-batch-gd'
      smb = SplitterMB.new(batch_size, train_data_x, train_data_y)
      train_data_x = smb.data_x
      train_data_y = smb.data_y
      @mb = 0
      while @mb < train_data_x.size
        @array_of_z = []
        @array_of_a = []

        @array_of_delta_a = []
        @array_of_delta_w = []
        @array_of_delta_b = []

        fit_forward(train_data_x[@mb], 0)
        i = 1
        while i < @array_of_layers.size - 1
          fit_forward(@array_of_z[i - 1], i)
          if i == @array_of_layers.size - 2
            puts 'Epoch: ' + (@mb).to_s + ' of: ' + (train_data_x.size - 1).to_s + ', train error: ' + apply_cost(cost_function, @array_of_a[i].flatten, train_data_y[@mb], i).to_s
          end
          i += 1
        end

        array_of_d = []
        i = 0
        while i < @array_of_a.size
          array_of_d[i] = []
          tmp = @g.random_matrix(@array_of_a[i].size, @array_of_a[i][0].size, 0.0..1.0)
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
            fit_backward_step_one(i, train_data_y[@mb], nil)
            i -= 1
          end
          i = @array_of_layers.size - 1
          while i > 1
            fit_backward_step_two(i - 1, alpha, nil)
            i -= 1
          end
        end
        @mb += 1
      end
    elsif optimizer == 'mini-batch-gd-w-momentum'
      smb = SplitterMB.new(batch_size, train_data_x, train_data_y)
      train_data_x = smb.data_x
      train_data_y = smb.data_y
      @array_of_v_delta_w = []
      @array_of_v_delta_b = []
      @mb = 0
      while @mb < train_data_x.size
        @array_of_z = []
        @array_of_a = []

        @array_of_delta_a = []
        @array_of_delta_w = []
        @array_of_delta_b = []

        fit_forward(train_data_x[@mb], 0)
        i = 1
        while i < @array_of_layers.size - 1
          fit_forward(@array_of_z[i - 1], i)
          if i == @array_of_layers.size - 2
            puts 'Epoch: ' + (@mb).to_s + ' of: ' + (train_data_x.size - 1).to_s + ', train error: ' + apply_cost(cost_function, @array_of_a[i].flatten, train_data_y[@mb], i).to_s
          end
          i += 1
        end

        array_of_d = []
        i = 0
        while i < @array_of_a.size
          array_of_d[i] = []
          tmp = @g.random_matrix(@array_of_a[i].size, @array_of_a[i][0].size, 0.0..1.0)
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
            fit_backward_step_one(i, train_data_y[@mb], momentum)
            i -= 1
          end
          i = @array_of_layers.size - 1
          while i > 1
            fit_backward_step_two(i - 1, alpha, momentum)
            i -= 1
          end
        end
        @mb += 1
      end
    end
    @array_of_a.last
  end

  def save_weights(path)
    serialized_array = Marshal.dump([@array_of_weights, @array_of_bias])
    File.open(path, 'wb') { |f| f.write(serialized_array) }
  end

  def save_architecture(path)
    serialized_array = Marshal.dump([@cost_function, @regularization_l2, @array_of_layers, @array_of_activations, @array_of_dropouts])
    File.open(path, 'wb') { |f| f.write(serialized_array) }
  end

  def load_weights(path)
    tmp = Marshal.load File.open(path, 'rb')
    @array_of_weights = tmp[0]
    @array_of_bias = tmp[1]
  end

  def load_architecture(path)
    tmp = Marshal.load File.open(path, 'rb')
    @cost_function = tmp[0]
    @regularization_l2 = tmp[1]
    @array_of_layers = tmp[2]
    layers = @array_of_layers.size
    nodes = @array_of_layers
    @array_of_layers = []
    @array_of_activations = tmp[3]
    @array_of_dropouts = tmp[4]

    i = 0
    while i < layers
      add_nn(nodes[i].size, @array_of_activations[i], @array_of_dropouts[i])
      i += 1
    end
  end

  def predict(dev_data_x, dev_data_y, cost_function, regularization_l2 = nil)
    @regularization_l2 = regularization_l2
    @array_of_z = []
    @array_of_a = []
    predict_forward(dev_data_x, 0)
    i = 1
    while i < @array_of_layers.size - 1
      predict_forward(@array_of_z[i - 1], i)
      if i == @array_of_layers.size - 2
        puts 'Prediction Error: ' + apply_cost(cost_function, @array_of_a[i].flatten, dev_data_y, i).to_s
      end
      i += 1
    end
    @array_of_a.last
  end

  private

  def create_weights(counter)
    @mm.mult(@g.random_matrix(@array_of_layers[counter], @array_of_layers[counter - 1], 0.0..0.01), Math.sqrt(2.0 / @features)) #/
  end

  def create_bias(counter)
    @g.zero_matrix(@array_of_layers[counter], @samples)
  end

  def fit_forward(z, counter)
    if counter.zero?
      z = z.transpose
    end
    @array_of_z[counter] = @mm.add(@mm.dot(@array_of_weights[counter], z), @array_of_bias[counter])
    @array_of_a[counter] = apply_a(@array_of_z[counter], counter + 1)
  end

  def predict_forward(z, counter)
    if counter.zero?
      z = z.transpose
    end
    @array_of_z[counter] = @mm.dot(@array_of_weights[counter], z)
    @array_of_a[counter] = apply_a(@array_of_z[counter], counter + 1)
  end

  def fit_backward_step_one(counter, data_y, momentum)
    if counter == @array_of_layers.size - 1
      if @cost_function == 'mse'
        @array_of_delta_a << @mm.subt(@array_of_a[counter - 1], data_y)
      elsif @cost_function == 'cross_entropy'
        tmp1 = @mm.mult(@mm.div(data_y, @array_of_a[counter - 1].flatten), -1.0)
        tmp2 = @mm.div(@mm.subt(data_y, 1.0), @mm.subt(@array_of_a[counter - 1].flatten, 1.0))
        @array_of_delta_a << [@mm.add(tmp1, tmp2)]
      end
    end
    delta_z = @mm.mult(@array_of_delta_a.pop, apply_d(@array_of_z[counter - 1], counter))
    @array_of_delta_a << @mm.dot(@array_of_weights[counter - 1].transpose, delta_z)
    if !@regularization_l2.nil?
      tmp = @mm.mult(@mm.dot(delta_z, @array_of_a[counter - 2].transpose), (1.0 / @samples)) #/
      @array_of_delta_w[counter - 2] = @mm.add(tmp, @mm.mult(@array_of_weights[counter - 1], (@regularization_l2 / @samples))) #/
    else
      @array_of_delta_w[counter - 2] = @mm.mult(@mm.dot(delta_z, @array_of_a[counter - 2].transpose), (1.0 / @samples)) #/
    end
    @array_of_delta_b[counter - 2] = @mm.mult(delta_z, (1.0 / @samples)) #/

    if !momentum.nil?
      if @mb.zero?
        @array_of_v_delta_w[counter - 2] = @g.zero_matrix(@array_of_delta_w[counter - 2].size, @array_of_delta_w[counter - 2][0].size)
        @array_of_v_delta_b[counter - 2] = @g.zero_matrix(@array_of_delta_b[counter - 2].size, @array_of_delta_b[counter - 2][0].size)
      elsif @mb == 1
        @array_of_v_delta_w[counter - 2] = @mm.mult(@array_of_v_delta_w[counter - 2], (1.0 / (momentum**@mb)))
        @array_of_v_delta_b[counter - 2] = @mm.mult(@array_of_v_delta_b[counter - 2], (1.0 / (momentum**@mb)))
      end

      tmp1 = @mm.mult(@array_of_v_delta_w[counter - 2], momentum)
      tmp2 = @mm.mult(@array_of_delta_w[counter - 2], (1.0 - momentum))
      @array_of_v_delta_w[counter - 2] = @mm.add(tmp1, tmp2)

      tmp1 = @mm.mult(@array_of_v_delta_b[counter - 2], momentum)
      tmp2 = @mm.mult(@array_of_delta_b[counter - 2], (1.0 - momentum))
      @array_of_v_delta_b[counter - 2] = @mm.add(tmp1, tmp2)
    end
  end

  def fit_backward_step_two(counter, alpha, momentum)
    if momentum.nil?
      @array_of_weights[counter] = @mm.subt(@array_of_weights[counter], @mm.mult(@array_of_delta_w[counter - 1], alpha))
      @array_of_bias[counter] = @mm.subt(@array_of_bias[counter], @mm.mult(@array_of_delta_b[counter - 1], alpha))
    else
      @array_of_weights[counter] = @mm.subt(@array_of_weights[counter], @mm.mult(@array_of_v_delta_w[counter - 1], alpha))
      @array_of_bias[counter] = @mm.subt(@array_of_bias[counter], @mm.mult(@array_of_v_delta_b[counter - 1], alpha))
    end
  end

  def apply_cost(cost_function, data_x, data_y, counter)
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
