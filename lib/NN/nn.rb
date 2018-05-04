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

  def add_nn(batch_size, activation, dropout = 1.0, batch_norm = nil)
    @array_of_layers << batch_size
    @array_of_activations << activation
    @array_of_dropouts << dropout
    @array_of_batch_norms << batch_norm
  end

  def compile(optimizer, cost_function, learning_rate = 0.000001, decay_rate = 1, iterations = 10, regularization_l2 = 0.1, momentum = [0.9, 0.999, 10**-8])
    @cost_function = cost_function
    @optimizer = optimizer
    @learning_rate = learning_rate
    @decay_rate = decay_rate
    @iterations = iterations
    @regularization_l2 = regularization_l2
    @momentum = momentum

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
        if time.size % 20 == 0
          time.shift(time.size / 2)
          clear = true
        end

        @array_of_a = []
        @array_of_z = []

        counter += 1
        create_layers(train_data_x)

        if mini_batch_samples.zero? || mini_batch_samples == 1 || clear
          puts 'Iter: ' + (counter * @iterations).to_s + ' of: ' + (epochs * train_data_x.size * @iterations).to_s + ', train error: ' + \
            apply_cost(train_data_y).to_s + ', ends: ' + '~' + ' minutes'
        else
          puts 'Iter: ' + (counter * @iterations).to_s + ' of: ' + (epochs * train_data_x.size * @iterations).to_s + ', train error: ' + \
            apply_cost(train_data_y).to_s + ', ends: ' + clock.to_s + ' minutes'
        end

        apply_dropout

        create_delta_arrays

        @iterations.times do
          @array_of_delta_w = []
          @array_of_delta_b = []
          back_propagation(train_data_x[mini_batch_samples], train_data_y[mini_batch_samples], mini_batch_samples)
          update_weights
        end

        @end_time = Time.new
        mini_batch_samples += 1
      end
    end
    @array_of_a.last
  end

  def predict(dev_data_x, dev_data_y, batch_size)
    @array_of_z = []
    @array_of_a = []

    smb = SplitterMB.new(batch_size, dev_data_x, dev_data_y)
    dev_data_x = smb.data_x
    dev_data_y = smb.data_y

    @array_of_a = []
    @array_of_z = []

    create_layers(dev_data_x)

    puts 'Prediction error: ' + apply_cost(dev_data_y).to_s

    @array_of_a.last
  end

  def save_weights(path)
    serialized_array = Marshal.dump([@array_of_weights, @array_of_bias])
    File.open(path, 'wb') { |f| f.write(serialized_array) }
  end

  def save_architecture(path)
    serialized_array = Marshal.dump([@cost_function, @regularization_l2, @array_of_layers, @array_of_activations, @array_of_dropouts, @array_of_batch_norms])
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
    @array_of_batch_norms = tmp[5]

    i = 0
    while i < layers
      add_nn(nodes[i].size, @array_of_activations[i], @array_of_dropouts[i], @array_of_batch_norms[i])
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
        @array_of_a[layer][samples] = apply_activ(@array_of_z[layer][samples], @array_of_activations[layer + 1])
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

  def back_propagation(data_x, data_y, tim)
    delta_a = Array.new(@array_of_layers.size) { |e| e = [] }
    layer = @array_of_layers.size - 1
    while layer > 0
      features = 0
      while features < @features
        if @cost_function == 'mse'
          delta_a[layer][features] = @mm.subt(@array_of_a[layer - 1][features], [data_y].transpose) if layer == @array_of_layers.size - 1
          delta_z = @mm.mult(delta_a[layer][features], apply_deriv(@array_of_z[layer - 1][features], nil, @array_of_activations[layer]))
        elsif @cost_function == 'log_loss'
          if layer != @array_of_layers.size - 1
            delta_a[layer][features] = @mm.subt(@array_of_a[layer - 1][features], data_y) if layer == @array_of_layers.size - 1
            delta_z = @mm.mult(delta_a[layer][features], apply_deriv(@array_of_z[layer - 1][features], data_y, @array_of_activations[layer]))
          else
            delta_z = apply_deriv(@array_of_z[layer - 1][features], data_y, @array_of_activations[layer])
          end
        end
        delta_a[layer - 1][features] = @mm.dot(delta_z, @array_of_weights[layer - 1].transpose)
        if !@regularization_l2.nil?
          if layer - 2 >= 0
            tmp = @mm.mult(@mm.dot(@array_of_a[layer - 2][features].transpose, delta_z), (1.0 / data_x[0].size))
          else
            tmp = @mm.mult(@mm.dot(data_x.transpose, delta_z), (1.0 / data_x[0].size))
          end
          @array_of_delta_w[layer] = @mm.add(tmp, @mm.mult(@array_of_weights[layer - 1], (@regularization_l2 / data_x[0].size)))
        else
          if layer - 2 >= 0
            tmp = @mm.mult(@mm.dot(@array_of_a[layer - 2][features].transpose, delta_z), (1.0 / data_x[0].size))
          else
            tmp = @mm.mult(@mm.dot(data_x.transpose, delta_z), (1.0 / data_x[0].size))
          end
          @array_of_delta_w[layer] = tmp
        end
        @array_of_delta_b[layer] = @mm.mult(@mm.horizontal_sum(delta_z), (1.0 / data_x[0].size))
        features += 1
      end
      layer -= 1
    end
    layer = 1
    while layer < @array_of_delta_w.size
      if @optimizer == 'BGDwM'
        @array_of_v_delta_w[layer] = @g.zero_matrix(@array_of_delta_w[layer].size, @array_of_delta_w[layer][0].size)
        @array_of_v_delta_b[layer] = @g.zero_vector(@array_of_delta_b[layer].size)

        tmp1 = @mm.mult(@array_of_v_delta_w[layer], @momentum[0])
        tmp2 = @mm.mult(@array_of_delta_w[layer], (1.0 - @momentum[0]))
        @array_of_v_delta_w[layer] = @mm.add(tmp1, tmp2)
      elsif @optimizer == 'RMSprop'
        @array_of_s_delta_w[layer] = @g.zero_matrix(@array_of_delta_w[layer].size, @array_of_delta_w[layer][0].size)
        @array_of_s_delta_b[layer] = @g.zero_vector(@array_of_delta_b[layer].size)

        tmp1 = @mm.mult(@array_of_s_delta_w[layer], @momentum[0])
        tmp2 = @mm.mult(@mm.mult(@array_of_delta_w[layer], @array_of_delta_w[layer]), (1.0 - @momentum[0]))
        @array_of_s_delta_w[layer] = @mm.add(tmp1, tmp2)

        tmp1 = @mm.mult(@array_of_s_delta_b[layer], @momentum[0])
        tmp2 = @mm.mult(@mm.mult(@array_of_delta_b[layer], @array_of_delta_b[layer]), (1.0 - @momentum[0]))
        @array_of_s_delta_b[layer] = @mm.add(tmp1, tmp2)
      elsif @optimizer == 'Adam'
        @array_of_v_delta_w[layer] = @g.zero_matrix(@array_of_delta_w[layer].size, @array_of_delta_w[layer][0].size)
        @array_of_v_delta_b[layer] = @g.zero_vector(@array_of_delta_b[layer].size)
        @array_of_s_delta_w[layer] = @g.zero_matrix(@array_of_delta_w[layer].size, @array_of_delta_w[layer][0].size)
        @array_of_s_delta_b[layer] = @g.zero_vector(@array_of_delta_b[layer].size)

        tmp1 = @mm.mult(@array_of_v_delta_w[layer], @momentum[0])
        tmp2 = @mm.mult(@array_of_delta_w[layer], (1.0 - @momentum[0]))
        @array_of_v_delta_w[layer] = @mm.add(tmp1, tmp2)

        tmp1 = @mm.mult(@array_of_v_delta_b[layer], @momentum[0])
        tmp2 = @mm.mult(@array_of_delta_b[layer], (1.0 - @momentum[0]))
        @array_of_v_delta_b[layer] = @mm.add(tmp1, tmp2)

        tmp1 = @mm.mult(@array_of_s_delta_w[layer], @momentum[1])
        tmp2 = @mm.mult(@mm.mult(@array_of_delta_w[layer], @array_of_delta_w[layer]), (1.0 - @momentum[1]))
        @array_of_s_delta_w[layer] = @mm.add(tmp1, tmp2)

        tmp1 = @mm.mult(@array_of_s_delta_b[layer], @momentum[1])
        tmp2 = @mm.mult(@mm.mult(@array_of_delta_b[layer], @array_of_delta_b[layer]), (1.0 - @momentum[1]))
        @array_of_s_delta_b[layer] = @mm.add(tmp1, tmp2)

        tmp = (1.0 - (@momentum[0]**tim))
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

  def update_weights
    if @optimizer == 'BGD'
      layer = @array_of_layers.size - 1
      while layer > 0
        features = 0
        while features < @features
          @array_of_weights[layer - 1] = @mm.subt(@array_of_weights[layer - 1], @mm.mult(@array_of_delta_w[layer], @learning_rate))
          @array_of_bias[layer - 1] = @mm.subt(@array_of_bias[layer - 1], @mm.mult(@array_of_delta_b[layer], @learning_rate))
          features += 1
        end
        layer -= 1
      end
    elsif @optimizer == 'BGDwM'
      layer = @array_of_layers.size - 1
      while layer > 0
        features = 0
        while features < @features
          @array_of_weights[layer - 1] = @mm.subt(@array_of_weights[layer - 1], @mm.mult(@array_of_v_delta_w[layer], @learning_rate))
          @array_of_bias[layer - 1] = @mm.subt(@array_of_bias[layer - 1], @mm.mult(@array_of_v_delta_b[layer], @learning_rate))
          features += 1
        end
        layer -= 1
      end
    elsif @optimizer == 'RMSprop'
      layer = @array_of_layers.size - 1
      while layer > 0
        features = 0
        while features < @features
          tmp1 = @mm.matrix_sqrt(@array_of_s_delta_w[layer])
          tmp2 = @mm.div(@array_of_delta_w[layer], tmp1)
          @array_of_weights[layer - 1] = @mm.subt(@array_of_weights[layer - 1], @mm.mult(tmp2, @learning_rate))

          tmp1 = @mm.vector_sqrt(@array_of_s_delta_b[layer])
          tmp2 = @mm.div(@array_of_delta_b[layer], tmp1)
          @array_of_bias[layer - 1] = @mm.subt(@array_of_bias[layer - 1], @mm.mult(tmp2, @learning_rate))
          features += 1
        end
        layer -= 1
      end
    elsif @optimizer == 'Adam'
      layer = @array_of_layers.size - 1
      while layer > 0
        features = 0
        while features < @features
          tmp1 = @mm.matrix_sqrt(@array_of_s_delta_w[layer])
          tmp2 = @mm.add(@mm.div(@array_of_v_delta_w[layer], tmp1), @momentum[2])
          @array_of_weights[layer - 1] = @mm.subt(@array_of_weights[layer - 1], @mm.mult(tmp2, @learning_rate))

          tmp1 = @mm.vector_sqrt(@array_of_s_delta_b[layer])
          tmp2 = @mm.add(@mm.div(@array_of_v_delta_b[layer], tmp1), @momentum[2])
          @array_of_bias[layer - 1] = @mm.subt(@array_of_bias[layer - 1], @mm.mult(tmp2, @learning_rate))
          features += 1
        end
        layer -= 1
      end
    end
  end

  def apply_cost(data_y)
      tmp2 = @mm.f_norm(@array_of_weights.last)
      if @cost_function == 'mse'
        if !@regularization_l2.nil?
          tmp1 = @c.quadratic_cost_with_r(@array_of_a.last, data_y, @regularization_l2, tmp2)
        else
          tmp1 = @c.quadratic_cost(@array_of_a.last, data_y)
        end
      elsif @cost_function == 'log_loss'
        if !@regularization_l2.nil?
          tmp1 = @c.log_loss_cost_with_r(@array_of_a.last, data_y, @regularization_l2, tmp2)
        else
          tmp1 = @c.log_loss_cost(@array_of_a.last, data_y)
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
