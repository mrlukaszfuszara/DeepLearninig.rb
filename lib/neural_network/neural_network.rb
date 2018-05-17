class NeuralNetwork
  def initialize
    @mm = MatrixMath.new
    @g = Generators.new
    @a = Activations.new
    @c = Costs.new

    @array_of_layers = []
    @array_of_activations = []
    @array_of_dropouts = []
    @array_of_steps = []
    
    @array_of_weights = []
  end

  def input(batch_size, activation)
    add_neuralnet(batch_size, activation)
  end

  def add_neuralnet(batch_size, activation, dropout = 1.0)
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
    
    i = 0
    while i < @array_of_layers.size - 1
      @array_of_weights << create_weights(i)
      i += 1
    end
  end

  def fit(train_data_x, train_data_y, batch_size, epochs, dev_data = nil)
    counter = 0
    epochs.times do |t|
      @learning_rate = @learning_rate / (1.0 + @decay_rate * t)
      element = 0
      while element < train_data_x.size
        create_layers(train_data_x[element])

        apply_dropout
        i = 0
        while i < @iterations
          create_deltas
          back_propagation(train_data_x[element], train_data_y[element])
          update_weights
          i += 1
        end

        p @array_of_a.last

        counter += 1

        str = 'Epoch: ' + t.to_s + ', iter: ' + (counter * @iterations).to_s + ', of: ' + (epochs * train_data_x.size * @iterations).to_s + ', train error: ' + \
          apply_cost(@array_of_a.last, train_data_y[element]).to_s

        puts str

        windows_size = IO.console.winsize[1].to_f - 20.0

        max_val = (epochs * train_data_x.size).to_f
        current_val = counter.to_f
        pg_bar = current_val / max_val

        puts '[' + '#' * (pg_bar * windows_size).floor + '*' * (windows_size - (pg_bar * windows_size)).floor + '] ' + (100 * pg_bar).floor.to_s + '%'
        element += 1
      end
    end
    @array_of_a.last
  end

  def predict(test_data_x, test_data_y, batch_size)
    tmp = VectorizeArray.new
    test_data_y = tmp.all(test_data_y)

    smb = SplitterMiniBatch.new(batch_size, test_data_x, test_data_y)
    test_data_x = smb.data_x

    create_layers(test_data_x.last)

    p test_data_y[0]
    p @array_of_a.last
  end

  def save_weights(path)
    serialized_array = Marshal.dump(@array_of_weights)
    File.open(path, 'wb') { |f| f.write(serialized_array) }
  end

  def save_architecture(path)
    serialized_array = Marshal.dump([@array_of_layers, @array_of_activations, @array_of_dropouts, @optimizer, @cost_function, @learning_rate, @decay_rate, @iterations, @momentum, @regularization_l2])
    File.open(path, 'wb') { |f| f.write(serialized_array) }
  end

  def load_weights(path)
    tmp = Marshal.load File.open(path, 'rb')

    @array_of_weights = tmp
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
    @iterations = tmp[7]
    @momentum = tmp[8]
    @regularization_l2 = tmp[9]
    
    i = 0
    while i < layers
      add_neuralnet(nodes[i].size, @array_of_activations[i], @array_of_dropouts[i])
      i += 1
    end
  end

  private

  def create_weights(counter)
    @mm.mult(@g.random_matrix(@array_of_layers[counter], @array_of_layers[counter + 1], 0.0..0.01), Math.sqrt(2.0 / @array_of_layers[counter])) #/
  end

  def create_deltas
    @array_of_v_delta_w = []
    @array_of_s_delta_w = []
    layer = 0
    while layer < @array_of_layers.size - 1
      @array_of_v_delta_w[layer] = @g.small_matrix(@array_of_layers[layer], @array_of_layers[layer + 1])
      @array_of_s_delta_w[layer] = @g.small_matrix(@array_of_layers[layer], @array_of_layers[layer + 1])
      layer += 1
    end
  end

  def create_layers(data_x)
    @array_of_a = []
    @array_of_z = []

    layer = 0
    while layer < @array_of_layers.size - 1
      if layer.zero?
        @array_of_z[layer] = @mm.dot(data_x, @array_of_weights[layer])
      end
      if !layer.zero?
        @array_of_z[layer] = @mm.dot(@array_of_a[layer - 1], @array_of_weights[layer])
      end
      @array_of_a[layer] = apply_activ(@array_of_z[layer], @array_of_activations[layer + 1])
      layer += 1
    end
  end

  def back_propagation(data_x, data_y)
    delta_z = []
    delta_w = []

    layer = @array_of_a.size - 1
    while layer >= 0
      if layer == @array_of_a.size - 1
        if @cost_function == 'mse'
          delta_z[layer] = @mm.subt(@array_of_a[layer], data_y)
        elsif @cost_function == 'crossentropy'
          delta_z[layer] = @mm.subt(data_y, @array_of_a[layer])
        end
      else
        w_dot_d = @mm.dot(delta_z[layer + 1], @array_of_weights[layer - 1].transpose)
        deriv = apply_deriv(@array_of_a[layer], @array_of_activations[layer + 1])
        delta_z[layer] = @mm.mult(w_dot_d, deriv)
      end
      if layer > 0
        delta_w[layer] = @mm.dot(@array_of_a[layer - 1].transpose, delta_z[layer])
      else
        delta_w[layer] = @mm.dot(data_x.transpose, delta_z[layer])
      end
      if !@regularization_l2.nil?
        @array_of_delta_w[layer] = @mm.mult(@mm.mult(delta_w[layer], (1.0 / @array_of_layers[layer])), (@regularization_l2 / data_x.size))
      else
        @array_of_delta_w[layer] = @mm.mult(delta_w[layer], (1.0 / @array_of_layers[layer]))
      end
      layer -= 1
    end
  end

  def update_weights
    if @optimizer == 'BGD'
      layer = @array_of_weights.size - 1
      while layer >= 0
        @array_of_weights[layer] = @mm.subt(@array_of_weights[layer], @mm.mult(@array_of_delta_w[layer], @learning_rate))
        layer -= 1
      end
    elsif @optimizer == 'BGDwM'
      layer = @array_of_weights.size - 1
      while layer >= 0
        tmp0 = @mm.mult(@array_of_v_delta_w[layer], @momentum[0])
        tmp1 = @mm.mult(@array_of_delta_w[layer], (1.0 - @momentum[0]))
        @array_of_v_delta_w[layer] = @mm.add(tmp0, tmp1)
        @array_of_weights[layer] = @mm.subt(@array_of_weights[layer], @mm.mult(@array_of_v_delta_w[layer], @learning_rate))

        layer -= 1
      end
    elsif @optimizer == 'RMSprop'
      layer = @array_of_weights.size - 1
      while layer >= 0
        tmp0 = @mm.mult(@array_of_s_delta_w[layer], @momentum[0])
        tmp1 = @mm.mult(@mm.matrix_square(@array_of_delta_w[layer]), (1.0 - @momentum[0]))
        @array_of_s_delta_w[layer] = @mm.add(tmp0, tmp1)
        tmp3 = @mm.div(@array_of_delta_w[layer], @mm.matrix_sqrt(@array_of_s_delta_w[layer]))
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
        tmp1 = @mm.mult(@mm.matrix_square(@array_of_delta_w[layer]), (1.0 - @momentum[1]))
        @array_of_s_delta_w[layer] = @mm.add(tmp0, tmp1)
        tmp3 = @mm.div(@array_of_v_delta_w[layer], @mm.matrix_sqrt(@array_of_s_delta_w[layer]))

        @array_of_weights[layer] = @mm.subt(@array_of_weights[layer], @mm.mult(tmp3, @learning_rate))

        layer -= 1
      end
    end
  end

  def apply_dropout
    layer = 0
    while layer < @array_of_layers.size - 1
      if @array_of_dropouts[layer + 1] != 1.0
        array_of_dropouts_final = []
        array_of_random_values = @g.random_matrix(@array_of_a[layer].size, @array_of_a[layer][0].size, 0.0..1.0)
        sample = 0
        while sample < @array_of_a[layer].size
          array_of_dropouts_final[sample] = []
          nodes = 0
          while nodes < @array_of_a[layer][sample].size
            if array_of_random_values[sample][nodes] > @array_of_dropouts[layer + 1]
              array_of_dropouts_final[sample][nodes] = 0.0
            else
              array_of_dropouts_final[sample][nodes] = 1.0
            end
            nodes += 1
          end
          sample += 1
        end
        @array_of_a[layer] = @mm.mult(@mm.mult(@array_of_a[layer], array_of_dropouts_final), (1.0 / @array_of_dropouts[layer + 1]))
      end
      layer += 1
    end
  end

  def apply_cost(last_layer, data_y)
    tmp2 = @mm.f_norm(@array_of_weights.last)
    if @cost_function == 'mse'
      if !@regularization_l2.nil?
        tmp1 = @c.mse_cost(last_layer, data_y, @regularization_l2, tmp2)
      else
        tmp1 = @c.mse_cost(last_layer, data_y)
      end
    elsif @cost_function == 'crossentropy'
      if !@regularization_l2.nil?
        tmp1 = @c.crossentropy_cost(last_layer, data_y, @regularization_l2, tmp2)
      else
        tmp1 = @c.crossentropy_cost(last_layer, data_y)
      end
    end
    tmp1
  end

  def apply_activ(layer, activation)
    if activation == 'relu'
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
