class NN
  def initialize(data_x_vertical_size)
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

    @features = data_x_vertical_size

    add_nn(@features, 'nil')
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

  def fit(train_data_x, train_data_y, batch_size, epochs = 10)
    smb = SplitterMB.new(batch_size, train_data_x, train_data_y)
    train_data_x = smb.data_x
    train_data_y = smb.data_y

    @samples = train_data_x.size

    counter = 0
    epochs.times do |t|
      @learning_rate = @learning_rate / (1.0 + @decay_rate * t)

      time = []

      mini_batch_samples = 0
      while mini_batch_samples < train_data_x.size
        @start_time = Time.new
        
        time << ((epochs * train_data_x.size) - (counter)) * (@start_time - @end_time) * 1_000_000 if mini_batch_samples > 0

        clock = (1.0 + time.inject(:+) / time.size / 60.0).floor if mini_batch_samples > 1

        clear = false
        if time.size % 20 == 0 || mini_batch_samples == 1
          time.shift(time.size / 2)
          clear = true
        end

        counter += 1
        create_layers(train_data_x)

        last_layer = @array_of_a.last[mini_batch_samples]

        if clear
          puts 'Iter: ' + (counter * @iterations).to_s + ' of: ' + (epochs * train_data_x.size * @iterations).to_s + ', train error: ' + \
            apply_cost(last_layer, train_data_y).to_s + ', ends: ' + '~' + ' minutes'
        else
          puts 'Iter: ' + (counter * @iterations).to_s + ' of: ' + (epochs * train_data_x.size * @iterations).to_s + ', train error: ' + \
            apply_cost(last_layer, train_data_y).to_s + ', ends: ' + clock.to_s + ' minutes'
        end

        apply_dropout

        create_delta_arrays

        @iterations.times do
          @array_of_delta_w = []
          @array_of_delta_b = []
          back_propagation(train_data_x[mini_batch_samples], train_data_y[mini_batch_samples], mini_batch_samples, batch_size)
          update_weights
        end

        @end_time = Time.new
        mini_batch_samples += 1
      end
    end
    @array_of_a.last
  end

  def predict(dev_data_x, dev_data_y, batch_size, index_of_parameter)
    smb = SplitterMB.new(batch_size, dev_data_x, dev_data_y)
    dev_data_x = smb.data_x
    dev_data_y = smb.data_y

    @samples = dev_data_x.size

    @array_of_a = []
    @array_of_z = []

    create_layers(dev_data_x)

    prec = 0
    rec = 0

    i = 0
    while i < dev_data_x.size
      prec += apply_precision(@array_of_a.last, dev_data_y, index_of_parameter)
      rec += apply_recall(@array_of_a.last, dev_data_y, index_of_parameter)
      i += 1
    end

    puts 'Accurency F1 Score: ' + f1_score(prec.to_f / dev_data_x.size, rec.to_f / dev_data_x.size)
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
    
    i = 0
    while i < layers
      add_nn(nodes[i].size, @array_of_activations[i], @array_of_dropouts[i])
      i += 1
    end
  end

  private

  def create_weights(counter)
    @mm.mult(@g.random_matrix(@array_of_layers[counter], @array_of_layers[counter + 1], 0.001..0.01), Math.sqrt(2.0 / @features)) #/
  end

  def create_bias(counter)
    @g.zero_vector(@array_of_layers[counter + 1])
  end

  def create_layers(data_x)
    @array_of_a = []
    @array_of_z = []

    layer = 0
    while layer < @array_of_layers.size - 1
      @array_of_z[layer] = []
      @array_of_a[layer] = []
      mini_batch_samples = 0
      while mini_batch_samples < data_x.size
        @array_of_z[layer][mini_batch_samples] = []
        @array_of_a[layer][mini_batch_samples] = []
        if layer.zero?
          @array_of_z[layer][mini_batch_samples] = @mm.add(@mm.dot(data_x[mini_batch_samples], @array_of_weights[layer]), @array_of_bias[layer])
        else
          @array_of_z[layer][mini_batch_samples] = @mm.add(@mm.dot(@array_of_a[layer - 1][mini_batch_samples], @array_of_weights[layer]), @array_of_bias[layer])
        end
        @array_of_a[layer][mini_batch_samples] = apply_activ(@array_of_z[layer][mini_batch_samples], @array_of_activations[layer + 1])
        mini_batch_samples += 1
      end
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
      @array_of_v_delta_w[layer] = @g.zero_matrix(@array_of_layers[layer], @array_of_layers[layer + 1])
      @array_of_v_delta_b[layer] = @g.zero_vector(@array_of_layers[layer + 1])
      @array_of_s_delta_w[layer] = @g.zero_matrix(@array_of_layers[layer], @array_of_layers[layer + 1])
      @array_of_s_delta_b[layer] = @g.zero_vector(@array_of_layers[layer + 1])
      layer += 1
    end
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
            if array_of_random_values[mini_batch_samples][nodes] <= @array_of_dropouts[layer]
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

  def back_propagation(data_x, data_y, tim, bs)
    delta_a = []
    layer = @array_of_layers.size - 1
    while layer > 0
      features = 0
      while features < @features
        if @cost_function == 'mse'
          delta_a << @mm.subt(@array_of_a[layer - 1][features], [data_y].transpose) if layer == @array_of_layers.size - 1
          delta_z = @mm.mult(delta_a.pop, apply_deriv(@array_of_z[layer - 1][features], nil, @array_of_activations[layer]))
          delta_a.unshift @mm.dot(delta_z, @array_of_weights[layer - 1].transpose)
        elsif @cost_function == 'log_loss'
          delta_a << apply_deriv(@array_of_z[layer - 1][features], data_y, 'softmax') if layer == @array_of_layers.size - 1
          delta_z = @mm.mult(delta_a.pop, apply_deriv(@array_of_z[layer - 1][features], data_y, 'softmax'))
          delta_a.unshift @mm.dot(delta_z, @array_of_weights[layer - 1].transpose)
        end
        if !@regularization_l2.nil?
          if layer - 2 >= 0
            tmp = @mm.mult(@mm.dot(@array_of_a[layer - 2][features].transpose, delta_z), (1.0 / bs))
          else
            tmp = @mm.mult(@mm.dot(data_x.transpose, delta_z), (1.0 / bs))
          end
          @array_of_delta_w[layer - 1] = @mm.add(tmp, @mm.mult(@array_of_weights[layer - 1], (@regularization_l2 / bs)))
        else
          if layer - 2 >= 0
            tmp = @mm.mult(@mm.dot(@array_of_a[layer - 2][features].transpose, delta_z), (1.0 / bs))
          else
            tmp = @mm.mult(@mm.dot(data_x.transpose, delta_z), (1.0 / bs))
          end
          @array_of_delta_w[layer - 1] = tmp
        end
        @array_of_delta_b[layer - 1] = @mm.mult(@mm.horizontal_sum(delta_z), (1.0 / bs))
        features += 1
      end
      layer -= 1
    end
    layer = 0
    while layer < @array_of_delta_w.size
      if @optimizer == 'BGDwM'
        tmp1 = @mm.mult(@array_of_v_delta_w[layer], @momentum[0])
        tmp2 = @mm.mult(@array_of_delta_w[layer], (1.0 - @momentum[0]))
        @array_of_v_delta_w[layer] = @mm.add(tmp1, tmp2)

        tmp1 = @mm.mult(@array_of_s_delta_b[layer], @momentum[0])
        tmp2 = @mm.mult(@array_of_delta_b[layer], (1.0 - @momentum[0]))
        @array_of_s_delta_b[layer] = @mm.add(tmp1, tmp2)
      elsif @optimizer == 'RMSprop'
        tmp1 = @mm.mult(@array_of_s_delta_w[layer], @momentum[0])
        tmp2 = @mm.mult(@mm.mult(@array_of_delta_w[layer], @array_of_delta_w[layer]), (1.0 - @momentum[0]))
        @array_of_s_delta_w[layer] = @mm.add(tmp1, tmp2)

        tmp1 = @mm.mult(@array_of_s_delta_b[layer], @momentum[0])
        tmp2 = @mm.mult(@mm.mult(@array_of_delta_b[layer], @array_of_delta_b[layer]), (1.0 - @momentum[0]))
        @array_of_s_delta_b[layer] = @mm.add(tmp1, tmp2)
      elsif @optimizer == 'Adam'
        tmp1 = @mm.mult(@array_of_v_delta_w[layer], @momentum[0])
        tmp2 = @mm.mult(@array_of_delta_w[layer], (1.0 - @momentum[0]))
        @array_of_v_delta_w[layer] = @mm.add(tmp1, tmp2)

        tmp1 = @mm.mult(@array_of_s_delta_b[layer], @momentum[0])
        tmp2 = @mm.mult(@array_of_delta_b[layer], (1.0 - @momentum[0]))
        @array_of_s_delta_b[layer] = @mm.add(tmp1, tmp2)

        tmp1 = @mm.mult(@array_of_s_delta_w[layer], @momentum[0])
        tmp2 = @mm.mult(@mm.mult(@array_of_delta_w[layer], @array_of_delta_w[layer]), (1.0 - @momentum[0]))
        @array_of_s_delta_w[layer] = @mm.add(tmp1, tmp2)

        tmp1 = @mm.mult(@array_of_s_delta_b[layer], @momentum[0])
        tmp2 = @mm.mult(@mm.mult(@array_of_delta_b[layer], @array_of_delta_b[layer]), (1.0 - @momentum[0]))
        @array_of_s_delta_b[layer] = @mm.add(tmp1, tmp2)
      end
      layer += 1
    end
  end

  def update_weights
    if @optimizer == 'BGD'
      layer = @array_of_layers.size - 2
      while layer > 0
        @array_of_weights[layer] = @mm.subt(@array_of_weights[layer], @mm.mult(@array_of_delta_w[layer], @learning_rate))
        @array_of_bias[layer] = @mm.subt(@array_of_bias[layer], @mm.mult(@array_of_delta_b[layer], @learning_rate))
        layer -= 1
      end
    elsif @optimizer == 'BGDwM'
      layer = @array_of_layers.size - 2
      while layer > 0
        @array_of_weights[layer] = @mm.subt(@array_of_weights[layer], @mm.mult(@array_of_v_delta_w[layer], @learning_rate))
        @array_of_bias[layer] = @mm.subt(@array_of_bias[layer], @mm.mult(@array_of_v_delta_b[layer], @learning_rate))
        layer -= 1
      end
    elsif @optimizer == 'RMSprop'
      layer = @array_of_layers.size - 2
      while layer > 0
        tmp1 = @mm.matrix_sqrt(@array_of_s_delta_w[layer])
        tmp2 = @mm.div(@array_of_delta_w[layer], tmp1)
        @array_of_weights[layer] = @mm.subt(@array_of_weights[layer], @mm.mult(tmp2, @learning_rate))

        tmp1 = @mm.vector_sqrt(@array_of_s_delta_b[layer])
        tmp2 = @mm.div(@array_of_delta_b[layer], tmp1)
        @array_of_bias[layer] = @mm.subt(@array_of_bias[layer], @mm.mult(tmp2, @learning_rate))
        layer -= 1
      end
    elsif @optimizer == 'Adam'
      layer = @array_of_layers.size - 2
      while layer > 0
        tmp1 = @mm.matrix_sqrt(@array_of_s_delta_w[layer])
        tmp2 = @mm.add(@mm.div(@array_of_v_delta_w[layer], tmp1), @momentum[2])
        @array_of_weights[layer] = @mm.subt(@array_of_weights[layer], @mm.mult(tmp2, @learning_rate))

        tmp1 = @mm.vector_sqrt(@array_of_s_delta_b[layer])
        tmp2 = @mm.add(@mm.div(@array_of_v_delta_b[layer], tmp1), @momentum[2])
        @array_of_bias[layer] = @mm.subt(@array_of_bias[layer], @mm.mult(tmp2, @learning_rate))
        layer -= 1
      end
    end
  end

  def apply_precision(last_layer, dev_data_y, ind)
    prec = 0
    mini_batch = 0
    while mini_batch < dev_data_y.size - 1
      sample = 0
      while sample < dev_data_y[mini_batch].size - 1
        check = dev_data_y[mini_batch][sample] - last_layer[mini_batch][sample][0]
        if check < 0.75 && check > -0.75 && dev_data_y[mini_batch][sample] == ind
          prec += 1
        end
        sample += 1
      end
      mini_batch += 1
    end
    prec.to_f / (dev_data_y.size * dev_data_y[mini_batch].size)
  end

  def apply_recall(last_layer, dev_data_y, ind)
    rec = 0
    mini_batch = 0
    while mini_batch < dev_data_y.size - 1
      sample = 0
      while sample < dev_data_y[mini_batch].size - 1
        check = last_layer[mini_batch][sample][0]
        if check < (0.75 + ind) && check > (-0.75 - ind)
          rec += 1
        end
        sample += 1
      end
      mini_batch += 1
    end
    rec.to_f / (dev_data_y.size * dev_data_y[mini_batch].size)
  end

  def f1_score(prec, rec)
    ((2.0 / ((1.0 / prec) + (1.0 / rec))) * 100).round(2).to_s + '%'
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
    if activation == 'relu'
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
