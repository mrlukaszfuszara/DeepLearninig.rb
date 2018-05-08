class NN
  def initialize
    @mm = MatrixMath.new
    @g = Generators.new
    @a = Activations.new
    @c = Costs.new

    @array_of_layers = []
    @array_of_activations = []
    @array_of_dropouts = []
    @array_of_batch_norms = []
    @array_of_weights = []
    @array_of_bias = []
  end

  def input(features, activation)
    @features = features
    add_nn(features, activation)
  end

  def add_nn(batch_size, activation, dropout = 1.0)
    @array_of_layers << batch_size
    @array_of_activations << activation
    @array_of_dropouts << dropout
  end

  def compile(optimizer, cost_function, learning_rate, decay_rate = 1, iterations = 10, momentum = [0.9, 0.999, 10**-8], regularization_l2 = 0.1)
    @cost_function = cost_function
    @optimizer = optimizer
    @learning_rate = learning_rate
    @decay_rate = decay_rate
    @iterations = iterations
    @momentum = momentum
    @regularization_l2 = regularization_l2

    @array_of_delta_w = []
    @array_of_delta_b = []
    create_delta_arrays

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

  def fit(train_data_x, train_data_y, batch_size, epochs, dev_data = nil)
    smb = SplitterMiniBatch.new(batch_size, train_data_x, train_data_y)
    train_data_x = smb.data_x
    train_data_y = smb.data_y

    if !dev_data.nil?
      smb = SplitterMiniBatch.new(batch_size, dev_data[0], dev_data[1])
      dev_data_x = smb.data_x
      dev_data_y = smb.data_y
      ind = dev_data[2]
    end

    @samples = train_data_x.size

    @tic = Time.new

    counter = 0
    epochs.times do |t|
      @learning_rate = @learning_rate / (1.0 + @decay_rate * t)

      time = []

      mini_batch_samples = 0
      while mini_batch_samples < train_data_x.size
        counter += 1

        @tac = Time.new
        
        time << ((epochs * train_data_x.size) - (counter)) * (@tac - @toc) * (@tac - @tic) * 1_000_000 * (1.0 / (@toc - @tic)) if mini_batch_samples > 0

        clock = (time.inject(:+) / time.size / 60.0).floor if mini_batch_samples > 1

        clear = false
        if time.size % 20 == 0 || mini_batch_samples == 1
          time.shift(time.size / 2)
          clear = true
        end

        create_layers(train_data_x[mini_batch_samples])
        apply_dropout
        i = 0
        while i < @iterations
          back_propagation(train_data_x[mini_batch_samples], train_data_y[mini_batch_samples])
          update_weights
          i += 1
        end
        create_layers(train_data_x[mini_batch_samples])

        if clear
          str = 'Epoch: ' + (t + 1).to_s + ', iter: ' + (counter * @iterations).to_s + ', of: ' + (epochs * train_data_x.size * @iterations).to_s + ', train error: ' + \
            apply_cost(@array_of_a.last, train_data_y[mini_batch_samples]).to_s + ', ends: ' + '~' + ' minutes'
        else
          str = 'Epoch: ' + (t + 1).to_s + ', iter: ' + (counter * @iterations).to_s + ', of: ' + (epochs * train_data_x.size * @iterations).to_s + ', train error: ' + \
            apply_cost(@array_of_a.last, train_data_y[mini_batch_samples]).to_s + ', ends: ' + clock.to_s + ' minutes'
        end

        puts str

        windows_size = IO.console.winsize[1].to_f

        max_val = (epochs * train_data_x.size * @iterations).to_f
        current_val = (counter).to_f
        pg_bar = current_val / max_val

        if pg_bar < 1.0
          puts '[' + '#' * (pg_bar * windows_size).floor + '*' * (windows_size - (pg_bar * windows_size + 20)).floor + '] ' + (pg_bar * 100).floor.to_s + '%'
        end

        @toc = Time.new
        mini_batch_samples += 1
      end
    end
    @array_of_a.last
  end

  def predict(dev_data_x, dev_data_y, batch_size, ind)
    smb = SplitterMiniBatch.new(batch_size, dev_data_x, dev_data_y)
    dev_data_x = smb.data_x
    dev_data_y = smb.data_y

    @samples = dev_data_x.size

    create_layers(dev_data_x.first)

    precision = apply_cost(@array_of_a.last, dev_data_y.first)

    puts 'Error: ' + (precision * 100).round(2).to_s + '%'
  end

  def save_weights(path)
    serialized_array = Marshal.dump([@array_of_weights, @array_of_bias])
    File.open(path, 'wb') { |f| f.write(serialized_array) }
  end

  def save_architecture(path)
    serialized_array = Marshal.dump([@array_of_layers, @array_of_activations, @array_of_dropouts, @optimizer, @cost_function, @learning_rate, @decay_rate, @iterations, @momentum, @regularization_l2])
    File.open(path, 'wb') { |f| f.write(serialized_array) }
  end

  def load_weights(path)
    tmp = Marshal.load File.open(path, 'rb')

    @array_of_weights = tmp[0]
    @array_of_bias = tmp[1]
  end

  def load_architecture(path)
    tmp = Marshal.load File.open(path, 'rb')

    @array_of_layers = tmp[0]
    layers = @array_of_layers.size
    nodes = @array_of_layers
    @array_of_layers = []

    @array_of_activations = tmp[1]
    @array_of_dropouts = tmp[2]

    @optimizer = tmp[3]
    @cost_function = tmp[4]
    @learning_rate = tmp[5]
    @decay_rate = tmp[6]
    @iterations = tmp[6]
    @momentum = tmp[8]
    @regularization_l2 = tmp[9]

    @features = nodes.first.size
    
    i = 0
    while i < layers
      add_nn(nodes[i].size, @array_of_activations[i], @array_of_dropouts[i])
      i += 1
    end
  end

  private

  def create_weights(counter)
    @mm.mult(@g.random_matrix(@array_of_layers[counter], @array_of_layers[counter + 1], 0.0..0.1), Math.sqrt(2.0 / @array_of_layers[counter])) #/
  end

  def create_bias(counter)
    @g.zero_vector(@array_of_layers[counter + 1])
  end

  def create_layers(data_x)
    @array_of_a = []
    @array_of_z = []

    layer = 0
    while layer < @array_of_layers.size - 1
      if layer.zero?
        @array_of_z[layer] = @mm.add(@mm.dot(data_x, @array_of_weights[layer]), @array_of_bias[layer])
      else
        @array_of_z[layer] = @mm.add(@mm.dot(@array_of_a[layer - 1], @array_of_weights[layer]), @array_of_bias[layer])
      end
      @array_of_a[layer] = apply_activ(@array_of_z[layer], @array_of_activations[layer])
      layer += 1
    end
  end

  def create_delta_arrays
    @array_of_v_delta_w = []
    @array_of_v_delta_b = []
    @array_of_s_delta_w = []
    @array_of_s_delta_b = []
    layer = 0
    while layer < @array_of_layers.size - 1
      @array_of_v_delta_w[layer] = @g.one_matrix(@array_of_layers[layer], @array_of_layers[layer + 1])
      @array_of_s_delta_w[layer] = @g.one_matrix(@array_of_layers[layer], @array_of_layers[layer + 1])
      layer += 1
    end
    layer = 0
    while layer < @array_of_layers.size - 1
      @array_of_v_delta_b[layer] = @g.zero_vector(@array_of_layers[layer + 1])
      @array_of_s_delta_b[layer] = @g.zero_vector(@array_of_layers[layer + 1])
      layer += 1
    end
  end

  def apply_dropout
    layer = 0
    while layer < @array_of_layers.size - 1
      array_of_dropouts_final = []
      array_of_random_values = @g.random_matrix(@array_of_a[layer].size, @array_of_a[layer][0].size, 0.0..1.0)
      sample = 0
      while sample < @array_of_a[layer].size
        array_of_dropouts_final[sample] = []
        nodes = 0
        while nodes < @array_of_a[layer][sample].size
          if array_of_random_values[sample][nodes] > @array_of_dropouts[layer]
            array_of_dropouts_final[sample][nodes] = 0.0
          else
            array_of_dropouts_final[sample][nodes] = 1.0
          end
          nodes += 1
        end
        sample += 1
      end
      @array_of_a[layer] = @mm.mult(@mm.mult(@array_of_a[layer], array_of_dropouts_final), (1.0 / @array_of_dropouts[layer])) #/
      layer += 1
    end
  end

  def back_propagation(data_x, data_y)
    layer = @array_of_a.size - 1
    while layer >= 0
      if @cost_function == 'mse'
        if layer == @array_of_a.size - 1
          delta_z = @mm.subt(@array_of_a[layer], [data_y].transpose)
          delta_w = @mm.dot(@array_of_a[layer - 1].transpose, delta_z)
        elsif layer > 0
          w_dot_d = @mm.dot(delta_z, @array_of_weights[layer + 1].transpose)
          deriv = apply_deriv(@array_of_a[layer], nil, @array_of_activations[layer])
          delta_z = @mm.mult(w_dot_d, deriv)
          delta_w = @mm.dot(@array_of_a[layer - 1].transpose, delta_z)
        elsif layer.zero?
          w_dot_d = @mm.dot(delta_z, @array_of_weights[layer + 1].transpose)
          deriv = apply_deriv(@array_of_a[layer], nil, @array_of_activations[layer])
          delta_z = @mm.mult(w_dot_d, deriv)
          delta_w = @mm.dot(data_x.transpose, delta_z)
        end
      elsif @cost_function == 'log_loss'
        if layer == @array_of_layers.size - 1
          delta_z = @mm.subt(@array_of_a[layer - 1], [data_y].transpose)
          delta_w = @mm.dot(@array_of_a[layer - 2].transpose, delta_z)
        else
          w_dot_d = @mm.dot(delta_z, @array_of_weights[layer].transpose)
          deriv = apply_deriv(@array_of_a[layer - 1].transpose, nil, @array_of_activations[layer])
          delta_z = @mm.mult(w_dot_d, deriv.transpose)
          delta_w = @mm.dot(@array_of_a[layer - 2].transpose, delta_z)
        end
      end
      if !@regularization_l2.nil?
        @array_of_delta_w[layer] = @mm.mult(@mm.mult(delta_w, (1.0 / data_x.size)), (@regularization_l2 / data_x.size))
        @array_of_delta_b[layer] = @mm.mult(@mm.mult(@mm.horizontal_sum(delta_w), (1.0 / data_x.size)), (@regularization_l2 / data_x.size))
      else
        @array_of_delta_w[layer] = @mm.mult(delta_w, (1.0 / data_x.size))
        @array_of_delta_b[layer] = @mm.mult(@mm.horizontal_sum(delta_w), (1.0 / data_x.size))
      end
      layer -= 1
    end
  end

  def update_weights
    if @optimizer == 'BGD'
      layer = @array_of_weights.size - 1
      while layer >= 0
        @array_of_weights[layer] = @mm.subt(@array_of_weights[layer], @mm.mult(@array_of_delta_w[layer], @learning_rate))
        @array_of_bias[layer] = @mm.subt(@array_of_bias[layer], @mm.mult(@array_of_delta_b[layer], @learning_rate))
        layer -= 1
      end
    elsif @optimizer == 'BGDwM'
      layer = @array_of_weights.size - 1
      while layer >= 0
        tmp0 = @mm.mult(@array_of_v_delta_w[layer], @momentum[0])
        tmp1 = @mm.mult(@array_of_delta_w[layer], (1.0 - @momentum[0]))
        @array_of_v_delta_w[layer] = @mm.add(tmp0, tmp1)
        @array_of_weights[layer] = @mm.subt(@array_of_weights[layer], @mm.mult(@array_of_v_delta_w[layer], @learning_rate))

        tmp0 = @mm.mult(@array_of_v_delta_b[layer], @momentum[0])
        tmp1 = @mm.mult(@array_of_delta_b[layer], (1.0 - @momentum[0]))
        @array_of_v_delta_b[layer] = @mm.add(tmp0, tmp1)
        @array_of_weights[layer] = @mm.subt(@array_of_weights[layer], @mm.mult(@array_of_v_delta_b[layer], @learning_rate))

        layer -= 1
      end
    elsif @optimizer == 'RMSprop'
      layer = @array_of_weights.size - 1
      while layer >= 0
        tmp0 = @mm.mult(@array_of_s_delta_w[layer], @momentum[0])
        tmp1 = @mm.mult(@mm.mult(@array_of_delta_w[layer], @array_of_delta_w[layer]), (1.0 - @momentum[0]))
        @array_of_s_delta_w[layer] = @mm.add(tmp0, tmp1)
        tmp3 = @mm.div(@array_of_delta_w[layer], @mm.matrix_sqrt(@array_of_s_delta_w[layer]))
        @array_of_weights[layer] = @mm.subt(@array_of_weights[layer], @mm.mult(tmp3, @learning_rate))

        tmp0 = @mm.mult(@array_of_s_delta_b[layer], @momentum[0])
        tmp1 = @mm.mult(@array_of_delta_b[layer], (1.0 - @momentum[0]))
        @array_of_s_delta_b[layer] = @mm.add(tmp0, tmp1)
        tmp3 = @mm.div(@array_of_delta_b[layer], @mm.vector_sqrt(@array_of_s_delta_b[layer]))
        @array_of_weights[layer] = @mm.subt(@array_of_weights[layer], @mm.mult(tmp3, @learning_rate))

        layer -= 1
      end
    elsif @optimizer == 'Adam'
      layer = @array_of_weights.size - 1
      while layer >= 0
        tmp0 = @mm.mult(@array_of_v_delta_w[layer], @momentum[0])
        tmp1 = @mm.mult(@array_of_delta_w[layer], (1.0 - @momentum[0]))
        @array_of_v_delta_w[layer] = @mm.add(tmp0, tmp1)

        tmp0 = @mm.mult(@array_of_s_delta_w[layer], @momentum[1])
        tmp1 = @mm.mult(@mm.mult(@array_of_delta_w[layer], @array_of_delta_w[layer]), (1.0 - @momentum[1]))
        @array_of_s_delta_w[layer] = @mm.add(tmp0, tmp1)
        tmp3 = @mm.div(@array_of_v_delta_w[layer], @mm.matrix_sqrt(@array_of_s_delta_w[layer]))

        @array_of_weights[layer] = @mm.subt(@array_of_weights[layer], @mm.mult(tmp3, @learning_rate))

        tmp0 = @mm.mult(@array_of_v_delta_b[layer], @momentum[0])
        tmp1 = @mm.mult(@array_of_delta_b[layer], (1.0 - @momentum[0]))
        @array_of_v_delta_b[layer] = @mm.add(tmp0, tmp1)

        tmp0 = @mm.mult(@array_of_s_delta_b[layer], @momentum[1])
        tmp1 = @mm.mult(@array_of_delta_b[layer], (1.0 - @momentum[1]))
        @array_of_s_delta_b[layer] = @mm.add(tmp0, tmp1)
        tmp3 = @mm.div(@array_of_v_delta_b[layer], @mm.vector_sqrt(@array_of_s_delta_b[layer]))

        @array_of_weights[layer] = @mm.subt(@array_of_weights[layer], @mm.mult(tmp3, @learning_rate))

        layer -= 1
      end
    end
  end

  def apply_cost(last_layer, data_y)
    tmp2 = @mm.f_norm(@array_of_weights.last)
    if @cost_function == 'mse'
      if !@regularization_l2.nil?
        tmp1 = @c.quadratic_cost_with_r(last_layer, data_y, @regularization_l2, tmp2)
      else
        tmp1 = @c.quadratic_cost(last_layer, data_y)
      end
    elsif @cost_function == 'log_loss'
      if !@regularization_l2.nil?
        tmp1 = @c.log_loss_cost_with_r(last_layer, data_y, @regularization_l2, tmp2)
      else
        tmp1 = @c.log_loss_cost(last_layer, data_y)
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
    elsif activation == 'softmax'
      tmp = @a.softmax(layer)
    end
    tmp
  end

  def apply_deriv(layer, hat, activation)
    if activation == 'nil'
      tmp = layer
    elsif activation == 'relu'
      tmp = @a.relu_d(layer)
    elsif activation == 'leaky_relu'
      tmp = @a.leaky_relu_d(layer)
    elsif activation == 'tanh'
      tmp = @a.tanh_d(layer)
    elsif activation == 'sigmoid'
      tmp = @a.sigmoid_d(layer)
    elsif activation == 'softmax'
      tmp = @a.softmax_d(layer, hat)
    end
    tmp
  end
end
