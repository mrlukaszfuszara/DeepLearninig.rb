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
    @array_of_bias = []
  end

  def input(batch_size, activation)
    @features = batch_size
    add_neuralnet(batch_size, activation)
  end

  def add_neuralnet(batch_size, activation, dropout = 1.0)
    @array_of_layers << batch_size
    @array_of_activations << activation
    @array_of_dropouts << dropout
  end

  def add_resnet(batch_size, step_size, total_steps, activation, dropout = 1.0)
    st = 0
    while st < total_steps
      @array_of_steps << ((@array_of_layers.size + 1)..(@array_of_layers.size + step_size - 1))
      i = 0
      while i < step_size
        @array_of_layers << batch_size
        @array_of_activations << activation
        @array_of_dropouts << dropout
        i += 1
      end
      st += 1
    end
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
    tmp = VectorizeArray.new
    train_data_y = tmp.all(train_data_y)

    smb = SplitterMiniBatch.new(batch_size, train_data_x, train_data_y)
    train_data_x = smb.data_x
    train_data_y = smb.data_y

    @tic = Time.new

    counter = 0
    epochs.times do |t|
      @learning_rate = @learning_rate / (1.0 + @decay_rate * t)

      time = []

      mini_batch_samples = 0
      while mini_batch_samples < train_data_x.size
        @tac = Time.new

        time << (epochs * train_data_x.size) / (@tac - @toc).to_f * (epochs * train_data_x.size - t * mini_batch_samples) if mini_batch_samples > 0

        clock = (time.inject(:+) / time.size / 1_000_000.0 / 60.0).round(2) if mini_batch_samples > 1

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

        windows_size = IO.console.winsize[1].to_f - 20.0

        counter += 1

        max_val = (epochs * train_data_x.size).to_f
        current_val = (counter).to_f
        pg_bar = current_val / max_val

        puts '[' + '#' * (pg_bar * windows_size).floor + '*' * (windows_size - (pg_bar * windows_size)).floor + '] ' + (100 * pg_bar).floor.to_s + '%'

        @toc = Time.new
        mini_batch_samples += 1
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
    serialized_array = Marshal.dump([@array_of_weights, @array_of_bias])
    File.open(path, 'wb') { |f| f.write(serialized_array) }
  end

  def save_architecture(path)
    serialized_array = Marshal.dump([@array_of_layers, @array_of_activations, @array_of_dropouts, @array_of_steps, @optimizer, @cost_function, @learning_rate, @decay_rate, @iterations, @momentum, @regularization_l2])
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
    @array_of_steps = tmp[3]

    @optimizer = tmp[4]
    @cost_function = tmp[5]
    @learning_rate = tmp[6]
    @decay_rate = tmp[7]
    @iterations = tmp[8]
    @momentum = tmp[9]
    @regularization_l2 = tmp[10]

    @features = nodes.first.size
    
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

  def create_bias(counter)
    @g.dotzeroone_vector(@array_of_layers[counter + 1])
  end

  def create_layers(data_x)
    @array_of_a = []
    @array_of_z = []

    layer = 0
    while layer < @array_of_layers.size - 1
      if layer.zero?
        @array_of_z[layer] = @mm.add(@mm.dot(data_x, @array_of_weights[layer]), @array_of_bias[layer])
      end
      if !layer.zero?
        @array_of_z[layer] = @mm.add(@mm.dot(@array_of_a[layer - 1], @array_of_weights[layer]), @array_of_bias[layer])
      end
      @array_of_a[layer] = apply_activ(@array_of_z[layer], @array_of_activations[layer + 1])
      layer += 1
    end
    block = 0
    while block < @array_of_steps.size
      step_min = @array_of_steps[block].first
      step_max = @array_of_steps[block].last
      times = step_max - step_min
      step = 0
      while step < times
        if step + step_min + 1 == step_max
          @array_of_z[step_max - 1] = @mm.add(@array_of_z[step_min], @array_of_a[step_max - 1])
        end
        step += 1
      end
      block += 1
    end
    layer = 0
    while layer < @array_of_layers.size - 1
      @array_of_a[layer] = apply_activ(@array_of_z[layer], @array_of_activations[layer + 1])
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
      @array_of_v_delta_b[layer] = @g.one_vector(@array_of_layers[layer + 1])
      @array_of_s_delta_b[layer] = @g.one_vector(@array_of_layers[layer + 1])
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
      @array_of_a[layer] = @mm.mult(@mm.mult(@array_of_a[layer], array_of_dropouts_final), (1.0 / @array_of_dropouts[layer + 1])) #/
      layer += 1
    end
  end

  def back_propagation(data_x, data_y)
    delta_z = []
    delta_w = []
    layer = @array_of_a.size - 1
    while layer >= 0
      if layer == @array_of_a.size - 1 && @cost_function == 'mse'
        delta_z[layer] = @mm.subt(@array_of_a[layer], data_y)
      elsif layer == @array_of_a.size - 1 && @cost_function == 'crossentropy'
        delta_z[layer] = @mm.subt(data_y, @array_of_a[layer])
      end
      if layer != @array_of_a.size - 1 && layer > 0
        w_dot_d = @mm.dot(delta_z[layer + 1], @array_of_weights[layer + 1].transpose)
        deriv = apply_deriv(@array_of_a[layer], nil, @array_of_activations[layer + 1])
        delta_z[layer] = @mm.mult(w_dot_d, deriv)
      end
      if layer.zero?
        w_dot_d = @mm.dot(delta_z[layer + 1], @array_of_weights[layer + 1].transpose)
        deriv = apply_deriv(@array_of_a[layer], nil, @array_of_activations[layer + 1])
        delta_z[layer] = @mm.mult(w_dot_d, deriv)
      end
      layer -= 1
    end
    block = 0
    while block < @array_of_steps.size
      step_min = @array_of_steps[block].first
      step_max = @array_of_steps[block].last
      times = step_max - step_min
      step = 0
      while step < times
        delta_z[step + step_min] = @mm.add(delta_z[step + step_min], @array_of_a[step + step_min])
        step += 1
      end
      block += 1
    end
    layer = @array_of_a.size - 1
    while layer >= 0
      if layer == @array_of_a.size - 1 && @cost_function == 'mse'
        delta_w[layer] = @mm.dot(@array_of_a[layer - 1].transpose, delta_z[layer])
      elsif layer == @array_of_a.size - 1 && @cost_function == 'crossentropy'
        delta_w[layer] = @mm.dot(@array_of_a[layer - 1].transpose, delta_z[layer])
      end
      if layer != @array_of_a.size - 1 && layer > 0
        delta_w[layer] = @mm.dot(@array_of_a[layer - 1].transpose, delta_z[layer])
      end
      if layer.zero?
        delta_w[layer] = @mm.dot(data_x.transpose, delta_z[layer])
      end
      layer -= 1
    end
    layer = @array_of_a.size - 1
    while layer >= 0
      if !@regularization_l2.nil?
        @array_of_delta_w[layer] = @mm.mult(@mm.mult(delta_w[layer], (1.0 / data_x.size)), (@regularization_l2 / data_x.size))
      else
        @array_of_delta_w[layer] = @mm.mult(delta_w[layer], (1.0 / data_x.size))
      end
      layer -= 1
    end
  end

  def update_weights
    skips = []
    block = 0
    while block < @array_of_steps.size
      skips << @array_of_steps[block].to_a
      block += 1
    end
    steps = 0
    while steps < skips.size
      skips[steps].pop
      steps += 1
    end
    skips = skips.flatten
    if @optimizer == 'BGD'
      layer = @array_of_weights.size - 1
      while layer >= 0
        if !skips.include?(layer)
          @array_of_weights[layer] = @mm.subt(@array_of_weights[layer], @mm.mult(@array_of_delta_w[layer], @learning_rate))
        end
        layer -= 1
      end
    elsif @optimizer == 'BGDwM'
      layer = @array_of_weights.size - 1
      while layer >= 0
        if !skips.include?(layer)
          tmp0 = @mm.mult(@array_of_v_delta_w[layer], @momentum[0])
          tmp1 = @mm.mult(@array_of_delta_w[layer], (1.0 - @momentum[0]))
          @array_of_v_delta_w[layer] = @mm.add(tmp0, tmp1)
          @array_of_weights[layer] = @mm.subt(@array_of_weights[layer], @mm.mult(@array_of_v_delta_w[layer], @learning_rate))
        end

        layer -= 1
      end
    elsif @optimizer == 'RMSprop'
      layer = @array_of_weights.size - 1
      while layer >= 0
        if !skips.include?(layer)
          tmp0 = @mm.mult(@array_of_s_delta_w[layer], @momentum[0])
          tmp1 = @mm.mult(@mm.mult(@array_of_delta_w[layer], @array_of_delta_w[layer]), (1.0 - @momentum[0]))
          @array_of_s_delta_w[layer] = @mm.add(tmp0, tmp1)
          tmp3 = @mm.div(@array_of_delta_w[layer], @mm.matrix_sqrt(@array_of_s_delta_w[layer]))
          @array_of_weights[layer] = @mm.subt(@array_of_weights[layer], @mm.mult(tmp3, @learning_rate))
        end

        layer -= 1
      end
    elsif @optimizer == 'Adam'
      layer = @array_of_weights.size - 1
      while layer >= 0
        if !skips.include?(layer)
          tmp0 = @mm.mult(@array_of_v_delta_w[layer], @momentum[0])
          tmp1 = @mm.mult(@array_of_delta_w[layer], (1.0 - @momentum[0]))
          @array_of_v_delta_w[layer] = @mm.add(tmp0, tmp1)

          tmp0 = @mm.mult(@array_of_s_delta_w[layer], @momentum[1])
          tmp1 = @mm.mult(@mm.mult(@array_of_delta_w[layer], @array_of_delta_w[layer]), (1.0 - @momentum[1]))
          @array_of_s_delta_w[layer] = @mm.add(tmp0, tmp1)
          tmp3 = @mm.div(@array_of_v_delta_w[layer], @mm.matrix_sqrt(@array_of_s_delta_w[layer]))

          @array_of_weights[layer] = @mm.subt(@array_of_weights[layer], @mm.mult(tmp3, @learning_rate))
        end

        layer -= 1
      end
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
