class NN
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
    i = 0
    while i < @array_of_layers.size - 1
      @array_of_bias << create_bias(i)
      i += 1
    end
    i = 0
    while i < @array_of_layers.size - 1
      @array_of_weights << create_weights(i)
      i += 1
    end
  end

  def fit(epochs, train_data_x, train_data_y, cost_function, optimizer, alpha, iterations, decay_rate, regularization_l2 = nil, batch_size = nil, momentum = [0.9, 0.999, 10**-8])
    @regularization_l2 = regularization_l2
    @cost_function = cost_function

    smb = SplitterMB.new(batch_size, train_data_x, train_data_y)
    train_data_x = smb.data_x
    train_data_y = smb.data_y

    counter = 0
    epochs.times do |t|
      alpha = alpha / (1.0 + decay_rate * t)

      time = []

      mini_batch_samples = 0
      while mini_batch_samples < train_data_x.size
        @start_time = Time.new
        
        time << ((epochs * train_data_x.size) - (counter)) * (@start_time - @end_time) * 1_000_000 if mini_batch_samples > 0

        clock = (1.0 + time.inject(:+) / time.size / 60.0).floor if mini_batch_samples > 1

        clear = false
        if time.size % 20 == 0
          time.shift(time.size / 2)
          clear = true
        end

        @array_of_a = []
        @array_of_z = []

        counter += 1
        create_layers(train_data_x)

        if mini_batch_samples.zero? || mini_batch_samples == 1 || clear
          puts 'Iter: ' + (counter * iterations).to_s + ' of: ' + (epochs * train_data_x.size * iterations).to_s + ', train error: ' + \
            apply_cost(cost_function, train_data_x.size, train_data_y[mini_batch_samples]).to_s + ', ends: ' + '~' + ' minutes'
        else
          puts 'Iter: ' + (counter * iterations).to_s + ' of: ' + (epochs * train_data_x.size * iterations).to_s + ', train error: ' + \
            apply_cost(cost_function, train_data_x.size, train_data_y[mini_batch_samples]).to_s + ', ends: ' + clock.to_s + ' minutes'
        end

        apply_dropout

        create_delta_arrays

        iterations.times do |r|
          @array_of_delta_w = []
          @array_of_delta_b = []
          back_propagation(train_data_x[mini_batch_samples], train_data_y[mini_batch_samples], momentum, optimizer, r)
          update_weights(alpha, momentum, optimizer)
        end

        @end_time = Time.new
        mini_batch_samples += 1
      end
    end
    @array_of_a.last
  end

  def predict(dev_data_x, dev_data_y, batch_size = nil)
    @array_of_z = []
    @array_of_a = []

    smb = SplitterMB.new(batch_size, dev_data_x, dev_data_y)
    dev_data_x = smb.data_x
    dev_data_y = smb.data_y

    mini_batch_samples = 0
      while mini_batch_samples < dev_data_x.size
        @array_of_a = []
        @array_of_z = []

        create_layers(dev_data_x)

        puts 'Prediction error: ' + apply_cost(@cost_function, dev_data_x.size, dev_data_y[mini_batch_samples]).to_s

        mini_batch_samples += 1
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

  private

  def create_weights(counter)
    @mm.mult(@g.random_matrix(@array_of_layers[counter], @array_of_layers[counter + 1], 0.0..0.01), Math.sqrt(2.0 / @features)) #/
  end

  def create_bias(counter)
    @g.zero_vector(@array_of_layers[counter + 1])
  end

  def create_layers(data_x)
    layer = 0
    while layer < @array_of_layers.size - 1
      @array_of_z[layer] = []
      @array_of_a[layer] = []
      samples = 0
      while samples < data_x.size
        if layer.zero?
          @array_of_z[layer][samples] = @mm.add(@mm.dot(data_x[samples], @array_of_weights[layer]), @array_of_bias[layer])
        else
          @array_of_z[layer][samples] = @mm.add(@mm.dot(@array_of_a[layer - 1][samples], @array_of_weights[layer]), @array_of_bias[layer])
        end
        @array_of_a[layer][samples] = apply_activ(@array_of_z[layer][samples], @array_of_activations[layer])
        samples += 1
      end
      layer += 1
    end
  end

  def create_delta_arrays
    @array_of_v_delta_w = []
    @array_of_v_delta_b = []
    @array_of_s_delta_w = []
    @array_of_s_delta_b = []
  end

  def apply_dropout
    layer = 0
    while layer < @array_of_layers.size - 1
      samples = 0
      while samples < @array_of_a[layer].size
        array_of_dropouts_final = []
        array_of_random_values = @g.random_matrix(@array_of_a[layer][samples].size, @array_of_a[layer][samples][0].size, 0.0..1.0)
        mini_batch_samples = 0
        while mini_batch_samples < @array_of_a[layer][0].size
          array_of_dropouts_final[mini_batch_samples] = []
          nodes = 0
          while nodes < @array_of_a[layer][0][0].size
            if array_of_random_values[mini_batch_samples][nodes] < @array_of_dropouts[layer]
              array_of_dropouts_final[mini_batch_samples][nodes] = 1.0
            else
              array_of_dropouts_final[mini_batch_samples][nodes] = 0.0
            end
            nodes += 1
          end
          mini_batch_samples += 1
        end
        @array_of_a[layer][samples] = @mm.mult(@mm.mult(@array_of_a[layer][samples], array_of_dropouts_final), (1.0 / @array_of_dropouts[layer])) #/
        samples += 1
      end
      layer += 1
    end
  end

  def back_propagation(data_x, data_y, momentum, optimizer, r)
    delta_a = Array.new(@array_of_layers.size) { |e| e = [] }
    layer = @array_of_layers.size - 1
    while layer > 0
      samples = 0
      while samples < data_y.size
        if @cost_function == 'mse'
          delta_a[layer][samples] = @mm.subt(@array_of_a[layer - 1][samples], [data_y].transpose) if layer == @array_of_layers.size - 1
          delta_z = @mm.mult(delta_a[layer][samples], apply_deriv(@array_of_z[layer - 1][samples], @array_of_activations[layer]))
          delta_a[layer - 1][samples] = @mm.dot(delta_z, @array_of_weights[layer - 1].transpose)
        end
        if !@regularization_l2.nil?
          if layer - 2 >= 0
            tmp = @mm.mult(@mm.dot(@array_of_a[layer - 2][samples].transpose, delta_z), (1.0 / @samples)) #/
          else
            tmp = @mm.mult(@mm.dot(data_x.transpose, delta_z), (1.0 / @samples)) #/
          end
          @array_of_delta_w[layer] = @mm.add(tmp, @mm.mult(@array_of_weights[layer - 1], (@regularization_l2 / @samples))) #/
        else
          if layer - 2 >= 0
            @array_of_delta_w[layer] = @mm.mult(@mm.dot(@array_of_a[layer - 2][samples].transpose, delta_z), (1.0 / @samples)) #/
          else
            @array_of_delta_w[layer] = @mm.mult(data_x.transpose, delta_z, (1.0 / @samples)) #/
          end
        end
        @array_of_delta_b[layer] = @mm.mult(@mm.horizontal_sum(delta_z), (1.0 / @samples)) #/
        samples += 1
      end
      layer -= 1
    end
    layer = 1
    while layer < @array_of_delta_w.size
      if optimizer == 'BGDwM'
        @array_of_v_delta_w[layer] = @g.zero_matrix(@array_of_delta_w[layer].size, @array_of_delta_w[layer][0].size)
        @array_of_v_delta_b[layer] = @g.zero_vector(@array_of_delta_b[layer].size)

        tmp1 = @mm.mult(@array_of_v_delta_w[layer], momentum[0])
        tmp2 = @mm.mult(@array_of_delta_w[layer], (1.0 - momentum[0]))
        @array_of_v_delta_w[layer] = @mm.add(tmp1, tmp2)
      elsif optimizer == 'RMSprop'
        @array_of_s_delta_w[layer] = @g.zero_matrix(@array_of_delta_w[layer].size, @array_of_delta_w[layer][0].size)
        @array_of_s_delta_b[layer] = @g.zero_vector(@array_of_delta_b[layer].size)

        tmp1 = @mm.mult(@array_of_s_delta_w[layer], momentum[0])
        tmp2 = @mm.mult(@mm.mult(@array_of_delta_w[layer], @array_of_delta_w[layer]), (1.0 - momentum[0]))
        @array_of_s_delta_w[layer] = @mm.add(tmp1, tmp2)

        tmp1 = @mm.mult(@array_of_s_delta_b[layer], momentum[0])
        tmp2 = @mm.mult(@mm.mult(@array_of_delta_b[layer], @array_of_delta_b[layer]), (1.0 - momentum[0]))
        @array_of_s_delta_b[layer] = @mm.add(tmp1, tmp2)
      elsif optimizer == 'Adam'
        @array_of_v_delta_w[layer] = @g.zero_matrix(@array_of_delta_w[layer].size, @array_of_delta_w[layer][0].size)
        @array_of_v_delta_b[layer] = @g.zero_vector(@array_of_delta_b[layer].size)
        @array_of_s_delta_w[layer] = @g.zero_matrix(@array_of_delta_w[layer].size, @array_of_delta_w[layer][0].size)
        @array_of_s_delta_b[layer] = @g.zero_vector(@array_of_delta_b[layer].size)

        tmp1 = @mm.mult(@array_of_v_delta_w[layer], momentum[0])
        tmp2 = @mm.mult(@array_of_delta_w[layer], (1.0 - momentum[0]))
        @array_of_v_delta_w[layer] = @mm.add(tmp1, tmp2)

        tmp1 = @mm.mult(@array_of_v_delta_b[layer], momentum[0])
        tmp2 = @mm.mult(@array_of_delta_b[layer], (1.0 - momentum[0]))
        @array_of_v_delta_b[layer] = @mm.add(tmp1, tmp2)

        tmp1 = @mm.mult(@array_of_s_delta_w[layer], momentum[1])
        tmp2 = @mm.mult(@mm.mult(@array_of_delta_w[layer], @array_of_delta_w[layer]), (1.0 - momentum[1]))
        @array_of_s_delta_w[layer] = @mm.add(tmp1, tmp2)

        tmp1 = @mm.mult(@array_of_s_delta_b[layer], momentum[1])
        tmp2 = @mm.mult(@mm.mult(@array_of_delta_b[layer], @array_of_delta_b[layer]), (1.0 - momentum[1]))
        @array_of_s_delta_b[layer] = @mm.add(tmp1, tmp2)

        tmp = (1.0 - (momentum[0]**r))
        if !tmp.zero?
          @array_of_v_delta_w[layer] = @mm.div(@array_of_v_delta_w[layer], tmp)
          @array_of_v_delta_b[layer] = @mm.div(@array_of_v_delta_b[layer], tmp)
          @array_of_s_delta_w[layer] = @mm.div(@array_of_s_delta_w[layer], tmp)
          @array_of_s_delta_b[layer] = @mm.div(@array_of_s_delta_b[layer], tmp)
        end
      end
      layer += 1
    end
  end

  def update_weights(alpha, momentum, optimizer)
    if optimizer == 'BGD'
      layer = @array_of_layers.size - 1
      while layer > 0
        samples = 0
        while samples < @array_of_a[layer - 1].size
          @array_of_weights[layer - 1] = @mm.subt(@array_of_weights[layer - 1], @mm.mult(@array_of_delta_w[layer], alpha))
          @array_of_bias[layer - 1] = @mm.subt(@array_of_bias[layer - 1], @mm.mult(@array_of_delta_b[layer], alpha))
          samples += 1
        end
        layer -= 1
      end
    elsif optimizer == 'BGDwM'
      layer = @array_of_layers.size - 1
      while layer > 0
        samples = 0
        while samples < @array_of_a[layer - 1].size
          @array_of_weights[layer - 1] = @mm.subt(@array_of_weights[layer - 1], @mm.mult(@array_of_v_delta_w[layer], alpha))
          @array_of_bias[layer - 1] = @mm.subt(@array_of_bias[layer - 1], @mm.mult(@array_of_v_delta_b[layer], alpha))
          samples += 1
        end
        layer -= 1
      end
    elsif optimizer == 'RMSprop'
      layer = @array_of_layers.size - 1
      while layer > 0
        samples = 0
        while samples < @array_of_a[layer - 1].size
          tmp1 = @mm.matrix_sqrt(@array_of_s_delta_w[layer])
          tmp2 = @mm.div(@array_of_delta_w[layer], tmp1)
          @array_of_weights[layer - 1] = @mm.subt(@array_of_weights[layer - 1], @mm.mult(tmp2, alpha))

          tmp1 = @mm.vector_sqrt(@array_of_s_delta_b[layer])
          tmp2 = @mm.div(@array_of_delta_b[layer], tmp1)
          @array_of_bias[layer - 1] = @mm.subt(@array_of_bias[layer - 1], @mm.mult(tmp2, alpha))
          samples += 1
        end
        layer -= 1
      end
    elsif optimizer == 'Adam'
      layer = @array_of_layers.size - 1
      while layer > 0
        samples = 0
        while samples < @array_of_a[layer - 1].size
          tmp1 = @mm.matrix_sqrt(@array_of_s_delta_w[layer])
          tmp2 = @mm.add(@mm.div(@array_of_v_delta_w[layer], tmp1), momentum[2])
          @array_of_weights[layer - 1] = @mm.subt(@array_of_weights[layer - 1], @mm.mult(tmp2, alpha))

          tmp1 = @mm.vector_sqrt(@array_of_s_delta_b[layer])
          tmp2 = @mm.add(@mm.div(@array_of_v_delta_b[layer], tmp1), momentum[2])
          @array_of_bias[layer - 1] = @mm.subt(@array_of_bias[layer - 1], @mm.mult(tmp2, alpha))
          samples += 1
        end
        layer -= 1
      end
        
    end
  end

  def apply_cost(cost_function, samples, data_y)
    if !@regularization_l2.nil?
      tmp2 = @mm.f_norm(@array_of_weights.last)
      if cost_function == 'mse'
        tmp1 = @c.quadratic_cost_with_r(@array_of_a.last.flatten, data_y, samples, @regularization_l2, tmp2)
      elsif cost_function == 'cross_entropy'
        tmp1 = @c.cross_entropy_cost_with_r(@array_of_a.last.flatten, data_y, samples, @regularization_l2, tmp2)
      end
    else
      if cost_function == 'mse'
        tmp1 = @c.quadratic_cost(@array_of_a.last.flatten, data_y, samples)
      elsif cost_function == 'cross_entropy'
        tmp1 = @c.cross_entropy_cost(@array_of_a.last.flatten, data_y, samples)
      end
    end
    tmp1
  end

  def apply_activ(layer, activation)
    if activation == 'nil'
      tmp = layer
    elsif activation == 'relu'
      tmp = @a.relu(layer)
    elsif activation == 'leaky_relu'
      tmp = @a.leaky_relu(layer)
    elsif activation == 'tanh'
      tmp = @a.tanh(layer)
    elsif activation == 'sigmoid'
      tmp = @a.sigmoid(layer)
    end
    tmp
  end

  def apply_deriv(layer, activation)
    if activation == 'relu'
      tmp = @a.relu_d(layer)
    elsif activation == 'leaky_relu'
      tmp = @a.leaky_relu_d(layer)
    elsif activation == 'tanh'
      tmp = @a.tanh_d(layer)
    elsif activation == 'sigmoid'
      tmp = @a.sigmoid_d(layer)
    end
    tmp
  end
end
