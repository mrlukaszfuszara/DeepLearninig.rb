require './lib/util/matrix_math'
require './lib/util/generators'
require './lib/util/activations'
require './lib/util/costs'

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

  def input(batch_size)
    add_neuralnet(batch_size, 'x')
  end

  def add_neuralnet(batch_size, activation, dropout = 1.0)
    @array_of_layers << batch_size
    @array_of_activations << activation
    @array_of_dropouts << dropout
  end

  def compile(optimizer, cost_function, learning_rate, decay_rate = 1, momentum = [0.9, 0.999, 10**-8])
    @cost_function = cost_function
    @optimizer = optimizer
    @learning_rate = learning_rate
    @decay_rate = decay_rate
    @momentum = momentum

    @array_of_delta_w = []

    i = 0
    while i < @array_of_layers.size - 1
      @array_of_weights << create_weights(i)
      i += 1
    end
  end

  def fit(train_data_x, train_data_y, epochs)
    counter = 0
    epochs.times do |t|
      create_deltas
      @learning_rate /= 1.0 + @decay_rate * t
      i = 0
      while i < train_data_x.size
        x = Matrix[*train_data_x[i]]
        y = Matrix[*train_data_y[i]]

        create_layers(x)

        stat = "Epoch: #{t}, of: #{epochs} epochs, iter: #{i}, of: #{train_data_x.size} iters, train error: #{apply_cost(@array_of_a.last, y)}"

        apply_dropout
        back_propagation(x, y)
        update_weights

        counter += 1

        windows_size = IO.console.winsize[1].to_f - 20.0

        max_val = (epochs * train_data_x.size).to_f
        current_val = counter.to_f
        pg_bar = current_val / max_val
        bar = '[' + '#' * (pg_bar * windows_size).floor + '*' * (windows_size - (pg_bar * windows_size)).floor + '] ' + (100 * pg_bar).floor.to_s + '%'

        puts stat
        puts bar

        i += 1
      end
    end
  end

  def predict(test_data_x, test_data_y)
    i = 0
    while i < test_data_x.size
      x = Matrix[*test_data_x[i]]
      y = Matrix[*test_data_y[i]]

      create_layers(x)

      p @array_of_a.last

      p 'Total cost: ' + apply_cost(@array_of_a.last, y).to_s

      i += 1
    end
  end

  def save_weights(path)
    serialized_array = Marshal.dump(@array_of_weights)
    File.open(path, 'wb') { |f| f.write(serialized_array) }
  end

  def save_architecture(path)
    serialized_array = Marshal.dump([@array_of_layers, @array_of_activations, @array_of_dropouts, @optimizer, @cost_function, @learning_rate, @decay_rate, @momentum])
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
    @momentum = tmp[7]

    i = 0
    while i < layers
      add_neuralnet(nodes[i].size, @array_of_activations[i], @array_of_dropouts[i])
      i += 1
    end
  end

  private

  def create_weights(counter)
    Matrix.build(@array_of_layers[counter + 1], @array_of_layers[counter]) { rand(0.0..0.01) * Math.sqrt(2.0 / @array_of_layers[counter]) }
  end

  def create_deltas
    @array_of_v_delta_w = []
    @array_of_s_delta_w = []
    layer = 0
    while layer < @array_of_layers.size - 1
      @array_of_v_delta_w << Matrix.build(@array_of_layers[layer + 1], @array_of_layers[layer]) { 1.0**-8 }
      @array_of_s_delta_w << Matrix.build(@array_of_layers[layer + 1], @array_of_layers[layer]) { 1.0**-8 }
      layer += 1
    end
  end

  def create_layers(data_x)
    @array_of_a = []
    @array_of_z = []

    layer = 0
    while layer < @array_of_layers.size
      if layer.zero?
        @array_of_z[layer] = data_x.transpose
        @array_of_a[layer] = @array_of_z[layer]
      elsif !layer.zero?
        @array_of_z[layer] = @array_of_weights[layer - 1] * @array_of_a[layer - 1]
        @array_of_a[layer] = apply_activ(@array_of_z[layer], @array_of_activations[layer])
      end
      layer += 1
    end
  end

  def back_propagation(data_x, data_y)
    delta = []

    layer = @array_of_layers.size - 1
    while layer > 0
      if layer == @array_of_layers.size - 1
        if @cost_function == 'crossentropy'
          delta[layer] = data_y.transpose - @array_of_a[layer]
        elsif @cost_function == 'mse'
          delta[layer] = data_y.transpose - @array_of_a[layer]
        end
      elsif layer != @array_of_layers.size - 1
        delta[layer] = (@array_of_weights[layer].transpose * delta[layer + 1]).hadamard_product(apply_deriv(@array_of_z[layer], @array_of_activations[layer]))
      end
      layer -= 1
    end
    layer = @array_of_layers.size - 1
    while layer > 0
      if layer != 1
        @array_of_delta_w[layer - 1] = (1.0 / @array_of_layers[layer]) * delta[layer] * @array_of_a[layer - 1].transpose
      elsif layer == 1
        @array_of_delta_w[layer - 1] = (1.0 / @array_of_layers[layer]) * delta[layer] * data_x
      end
      layer -= 1
    end
  end

  def update_weights
    if @optimizer == 'BGD'
      layer = @array_of_weights.size - 1
      while layer >= 0
        @array_of_weights[layer] -= @learning_rate * @array_of_delta_w[layer]
        layer -= 1
      end
    elsif @optimizer == 'BGDwM'
      layer = @array_of_weights.size - 1
      while layer >= 0
        @array_of_v_delta_w[layer] = @momentum[0] * @array_of_v_delta_w[layer] + (1.0 - @momentum[0]) * @array_of_delta_w[layer]
        @array_of_weights[layer] -= @learning_rate * @array_of_v_delta_w[layer]
        layer -= 1
      end
    elsif @optimizer == 'RMSprop'
      layer = @array_of_weights.size - 1
      while layer >= 0
        @array_of_s_delta_w[layer] = @momentum[0] * @array_of_s_delta_w[layer] + (1.0 - @momentum[0]) * @mm.matrix_square(@array_of_delta_w[layer])
        @array_of_weights[layer] -= @learning_rate * @mm.elementwise_div(@array_of_delta_w[layer], @mm.matrix_sqrt(@array_of_s_delta_w[layer]))
        layer -= 1
      end
    elsif @optimizer == 'Adam'
      layer = @array_of_weights.size - 1
      while layer >= 0
        @array_of_v_delta_w[layer] = @momentum[0] * @array_of_v_delta_w[layer] + (1.0 - @momentum[0]) * @array_of_delta_w[layer]
        @array_of_s_delta_w[layer] = @momentum[1] * @array_of_s_delta_w[layer] + (1.0 - @momentum[1]) * @mm.matrix_square(@array_of_delta_w[layer])
        @array_of_weights[layer] -= @learning_rate * @mm.elementwise_div(@array_of_v_delta_w[layer], @mm.elementwise_add(@mm.matrix_sqrt(@array_of_s_delta_w[layer]), @momentum[2]))
        layer -= 1
      end
    end
  end

  def apply_dropout
    layer = 0
    while layer < @array_of_a.size
      if @array_of_dropouts[layer] != 1.0
        random_values_matrix = Matrix.build(@array_of_a[layer].row_size, @array_of_a[layer].column_size) { rand(0.0..1.0) }
        row = 0
        while row < random_values_matrix.row_size
          col = 0
          while col < random_values_matrix.column_size
            if @array_of_dropouts[layer] > random_values_matrix[row, col]
              @array_of_a[layer][row, col] = 1.0 * @array_of_a[layer][row, col]
            else
              @array_of_a[layer][row, col] = 0.0 * @array_of_a[layer][row, col]
            end
            col += 1
          end
          row += 1
        end
      end
      layer += 1
    end
  end

  def apply_cost(last_layer, data_y)
    tmp = nil
    if @cost_function == 'mse'
      tmp = @c.mse_cost(last_layer, data_y.transpose)
    elsif @cost_function == 'crossentropy'
      tmp = @c.crossentropy_cost(last_layer, data_y.transpose)
    end
    tmp
  end

  def apply_activ(layer, activation)
    tmp = nil
    if activation == 'relu'
      tmp = @a.relu(layer)
    elsif activation == 'leaky_relu'
      tmp = @a.leaky_relu(layer)
    elsif activation == 'softmax'
      tmp = @a.softmax(layer)
    end
    tmp
  end

  def apply_deriv(layer, activation)
    tmp = nil
    if activation == 'relu'
      tmp = @a.relu_d(layer)
    elsif activation == 'leaky_relu'
      tmp = @a.leaky_relu_d(layer)
    end
    tmp
  end
end
