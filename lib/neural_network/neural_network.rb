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

  def input(batch_size, activation)
    add_neuralnet(batch_size, activation)
  end

  def add_neuralnet(batch_size, activation, dropout = 1.0)
    @array_of_layers << batch_size
    @array_of_activations << activation
    @array_of_dropouts << dropout
  end

  def compile(optimizer, cost_function, learning_rate, decay_rate = 1, iterations = 10, momentum = [0.9, 0.999, 10**-8])
    @cost_function = cost_function
    @optimizer = optimizer
    @learning_rate = learning_rate
    @decay_rate = decay_rate
    @iterations = iterations
    @momentum = momentum

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
      i = 0
      while i < train_data_x.size
        x = Matrix[*train_data_x[i]]
        y = Matrix[*train_data_y[i]]

        create_layers(x)

        apply_dropout
        j = 0
        while j < @iterations
          create_deltas
          back_propagation(x, y)
          update_weights
          j += 1
        end

        counter += 1

        windows_size = IO.console.winsize[1].to_f - 20.0

        str = 'Epoch: ' + t.to_s + ', of: ' + epochs.to_s + ' epochs, iter: ' + i.to_s + ', of: ' + train_data_x.size.to_s + ' iters, train error: ' + apply_cost(@array_of_a.last, y).to_s

        max_val = (epochs * train_data_x.size).to_f
        current_val = counter.to_f
        pg_bar = current_val / max_val

        puts str
        puts '[' + '#' * (pg_bar * windows_size).floor + '*' * (windows_size - (pg_bar * windows_size)).floor + '] ' + (100 * pg_bar).floor.to_s + '%'
        
        i += 1
      end
    end
  end

  def predict(test_data_x, test_data_y, batch_size)
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
    serialized_array = Marshal.dump([@array_of_layers, @array_of_activations, @array_of_dropouts, @optimizer, @cost_function, @learning_rate, @decay_rate, @iterations, @momentum])
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

    i = 0
    while i < layers
      add_neuralnet(nodes[i].size, @array_of_activations[i], @array_of_dropouts[i])
      i += 1
    end
  end

  private

  def create_weights(counter)
    Matrix.build(@array_of_layers[counter], @array_of_layers[counter + 1]) { rand(0.0..0.01) * Math.sqrt(2.0 / @array_of_layers[counter]) }
  end

  def create_deltas
    @array_of_v_delta_w = []
    layer = 0
    while layer < @array_of_layers.size - 1
      @array_of_v_delta_w << Matrix.build(@array_of_layers[layer], @array_of_layers[layer + 1]) { 1.0 }
      layer += 1
    end
  end

  def create_layers(data_x)
    @array_of_a = []
    @array_of_z = []

    layer = 0
    while layer < @array_of_layers.size
      if layer.zero?
        @array_of_z[layer] = data_x
        @array_of_a[layer] = data_x
      elsif !layer.zero?
        @array_of_z[layer] = @array_of_a[layer - 1] * @array_of_weights[layer - 1]
        @array_of_a[layer] = apply_activ(@array_of_z[layer], @array_of_activations[layer])
      end
      layer += 1
    end
  end

  def back_propagation(data_x, data_y)
    delta = []

    layer = @array_of_layers.size - 1
    while layer >= 0
      if layer == @array_of_layers.size - 1
        delta[layer] = data_y - @array_of_a[layer]
      elsif layer != @array_of_layers.size - 1
        delta[layer] = (delta[layer + 1] * @array_of_weights[layer].transpose).entrywise_product(apply_deriv(@array_of_z[layer], @array_of_activations[layer]))
      end
      layer -= 1
    end
    layer = @array_of_layers.size - 1
    while layer > 0
      @array_of_delta_w[layer - 1] = @array_of_a[layer].transpose * delta[layer - 1]
      layer -= 1
    end
  end

  def update_weights
    if @optimizer == 'BGD'
      layer = @array_of_weights.size - 1
      while layer >= 0
        @array_of_weights[layer] -= @learning_rate * @array_of_delta_w[layer].transpose
        layer -= 1
      end
    elsif @optimizer == 'BGDwM'
      layer = @array_of_weights.size - 1
      while layer >= 0
        tmp0 = @array_of_v_delta_w[layer] * @momentum[0]
        tmp1 = @array_of_delta_w[layer].transpose * (1.0 - @momentum[0])
        @array_of_v_delta_w[layer] = tmp0 + tmp1
        @array_of_weights[layer] -= @learning_rate * @array_of_v_delta_w[layer]
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
            @array_of_dropouts[layer] > random_values_matrix[row, col] ? @array_of_a[layer][row, col] * 1.0 : @array_of_a[layer][row, col] * 0.0
            col += 1
          end
          row += 1
        end
        @array_of_a[layer] = @array_of_a[layer] * (1.0 / @array_of_dropouts[layer])
      end
      layer += 1
    end
  end

  def apply_cost(last_layer, data_y)
    if @cost_function == 'mse'
      tmp1 = @c.mse_cost(last_layer, data_y)
    elsif @cost_function == 'crossentropy'
      tmp1 = @c.crossentropy_cost(last_layer, data_y)
    end
    tmp1
  end

  def apply_activ(layer, activation)
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
    if activation == 'relu'
      tmp = @a.relu_d(layer)
    elsif activation == 'leaky_relu'
      tmp = @a.leaky_relu_d(layer)
    end
    tmp
  end
end
