require 'digest'

require './lib/neural_network/network'

require './lib/util/matrix_math'
require './lib/util/generators'
require './lib/util/activations'
require './lib/util/costs'

class NeuralNetwork < Network
  def initialize
    @activation = Activations.new
    @cost = Costs.new

    @layers_array = []
    @activations_array = []
    @dropouts_array = []
    @weights_array = []
  end

  def input(batch_size, activation)
    add_neuralnet(batch_size, activation)
  end

  def add_neuralnet(batch_size, activation, dropout = 1.0)
    @layers_array << batch_size
    @activations_array << activation
    @dropouts_array << dropout
  end

  def compile(optimizer, cost_function, learning_rate, decay_rate = 1, momentum = [0.9, 0.999, 10**-8])
    @cost_function = cost_function
    @optimizer = optimizer
    @learning_rate = learning_rate
    @decay_rate = decay_rate
    @momentum = momentum

    @delta_w_array = []

    i = 0
    while i < @layers_array.size - 1
      @weights_array << create_weights(i)
      i += 1
    end
  end

  def fit(train_data_x, train_data_y, epochs, iterations)
    puts 'Lets start!'

    counter = 0
    epochs.times do |t|
      @learning_rate /= 1.0 + @decay_rate * t
      i = 0
      while i < train_data_x.size
        x = Matrix[*train_data_x[i]]
        y = Matrix[*train_data_y[i]]

        forward_propagation(x)
        stat = "Epoch: #{t}, of: #{epochs} epochs, iter: #{i}, of: #{train_data_x.size} iters in epoch, train error: #{apply_cost(y)}"

        apply_dropout

        create_deltas
        if iterations > 0
          print 'Optimisation: '
          j = 0
          while j < iterations
            backward_propagation(y)
            update_weights
            print '|>'
            j += 1
          end
          print "|\n"
        else
          backward_propagation(y)
          update_weights
        end

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

      forward_propagation(x)

      puts 'Total cost: ' + apply_cost(y).to_s

      i += 1
    end
  end

  def save_weights(path)
    save_data(path, @weights_array)
  end

  def save_architecture(path)
    save_data(path, [@layers_array, @activations_array, @dropouts_array, @optimizer, @cost_function, @learning_rate, @decay_rate, @momentum])
  end

  def load_weights(key, path)
    tmp = load_data(key, path)
    @weights_array = tmp
  end

  def load_architecture(key, path)
    tmp = load_data(key, path)

    @layers_array = tmp[0]
    layers = @layers_array.size
    nodes = @layers_array
    @layers_array = []

    @activations_array = tmp[1]
    @dropouts_array = tmp[2]

    @optimizer = tmp[3]
    @cost_function = tmp[4]
    @learning_rate = tmp[5]
    @decay_rate = tmp[6]
    @momentum = tmp[7]

    i = 0
    while i < layers
      add_neuralnet(nodes[i].size, @activations_array[i], @dropouts_array[i])
      i += 1
    end
  end

  private

  def create_weights(counter)
    Matrix.build(@layers_array[counter + 1], @layers_array[counter]) { rand(0.0..0.1) * Math.sqrt(2.0 / (@layers_array[counter + 1] + @layers_array[counter])) }
  end

  def create_deltas
    @v_delta_w_array = []
    @s_delta_w_array = []
    layer = 0
    while layer < @layers_array.size - 1
      @v_delta_w_array << Matrix.build(@layers_array[layer + 1], @layers_array[layer]) { 10**-8 }
      @s_delta_w_array << Matrix.build(@layers_array[layer + 1], @layers_array[layer]) { 10**-8 }
      layer += 1
    end
  end

  def forward_propagation(data_x)
    @a_array = []
    @z_array = []

    layer = 0
    while layer < @layers_array.size
      if layer.zero?
        @z_array[layer] = data_x
        @a_array[layer] = @z_array.clone[layer]
      else
        @z_array[layer] = @a_array.clone[layer - 1] * @weights_array.clone[layer - 1].transpose
        @a_array[layer] = apply_activ(@z_array.clone[layer], @activations_array[layer])
      end
      layer += 1
    end
  end

  def backward_propagation(data_y)
    if @cost_function == 'crossentropy'
      @delta_w_array = crossentropy_delta(data_y)
    elsif @cost_function == 'mse'
      @delta_w_array = mse_delta(data_y)
    end
  end

  def crossentropy_delta(data_y)
    delta_a = []
    delta_z = []
    delta_w = []
    layer = @layers_array.size - 1
    while layer > 0
      if layer == @layers_array.size - 1
        tmp = []
        i = 0
        while i < @z_array[layer].row_size
          tmp[i] = []
          j = 0
          while j < @z_array[layer].column_size
            if i == j
              tmp[i][j] = @a_array[layer][i, j] * (1.0 - @a_array[layer][i, j])
            else
              tmp[i][j] = -1.0 * @a_array[layer][i, j]**2
            end
            j += 1
          end
          i += 1
        end
        delta_a[layer] = (@a_array.clone[layer] - data_y).hadamard_product(Matrix[*tmp])
      else
        delta_a[layer] = delta_z.clone[layer + 1].hadamard_product(apply_deriv(@a_array.clone[layer], @activations_array[layer]))
      end
      delta_z[layer] = delta_a.clone[layer].hadamard_product(@mask_array.clone[layer]) * @weights_array.clone[layer - 1]
      delta_w[layer - 1] = @a_array.clone[layer].transpose * delta_z.clone[layer]
      layer -= 1
    end
    delta_w
  end

  def mse_delta(data_y)
    delta_a = []
    delta_z = []
    delta_w = []
    layer = @layers_array.size - 1
    while layer > 0
      if layer == @layers_array.size - 1
        delta_a[layer] = (data_y - @a_array.clone[layer]).hadamard_product(apply_deriv(@a_array.clone[layer], @activations_array[layer]))
      else
        delta_a[layer] = delta_z.clone[layer + 1].hadamard_product(apply_deriv(@a_array.clone[layer], @activations_array[layer]))
      end
      delta_z[layer] = delta_a.clone[layer].hadamard_product(@mask_array.clone[layer]) * @weights_array.clone[layer - 1]
      delta_w[layer - 1] = @a_array.clone[layer].transpose * delta_z.clone[layer]
      layer -= 1
    end
    delta_w
  end

  def update_weights
    if @optimizer == 'BGD'
      layer = @weights_array.size - 1
      while layer >= 0
        @weights_array[layer] -= @learning_rate * @delta_w_array.clone[layer]
        layer -= 1
      end
    elsif @optimizer == 'BGDwM'
      layer = @weights_array.size - 1
      while layer >= 0
        @v_delta_w_array[layer] = @momentum[0] * @v_delta_w_array.clone[layer] + (1.0 - @momentum[0]) * @delta_w_array.clone[layer]
        @weights_array[layer] -= @learning_rate * @v_delta_w_array.clone[layer]
        layer -= 1
      end
    elsif @optimizer == 'RMSprop'
      layer = @weights_array.size - 1
      while layer >= 0
        @s_delta_w_array[layer] = @momentum[0] * @s_delta_w_array.clone[layer] + (1.0 - @momentum[0]) * @delta_w_array.clone[layer].pow(2)
        @weights_array[layer] -= @learning_rate * @delta_w_array.clone[layer].elementwise_matrix_div(@s_delta_w_array.clone[layer].sqrt.elementwise_var_add(10**-8))
        layer -= 1
      end
    elsif @optimizer == 'Adam'
      layer = @weights_array.size - 1
      while layer >= 0
        @v_delta_w_array[layer] = @momentum[0] * @v_delta_w_array.clone[layer] + (1.0 - @momentum[0]) * @delta_w_array.clone[layer]
        @s_delta_w_array[layer] = @momentum[1] * @s_delta_w_array.clone[layer] + (1.0 - @momentum[1]) * @delta_w_array.clone[layer].pow(2)
        @v_delta_w_array[layer] = @v_delta_w_array.clone[layer].elementwise_var_div(1.0 - @momentum[0])
        @s_delta_w_array[layer] = @s_delta_w_array.clone[layer].elementwise_var_div(1.0 - @momentum[1])
        @weights_array[layer] -= @learning_rate * @v_delta_w_array.clone[layer].elementwise_matrix_div(@s_delta_w_array.clone[layer].sqrt.elementwise_var_add(@momentum[2]))
        layer -= 1
      end
    end
  end

  def apply_dropout
    @mask_array = []
    layer = 0
    while layer < @a_array.size
      @mask_array[layer] = Matrix.build(@a_array[layer].row_size, @a_array[layer].column_size) { 0.0 }
      row = 0
      while row < @a_array[layer].row_size
        column = 0
        while column < @a_array[layer].column_size
          if @dropouts_array[layer] >= rand(0.0..1.0)
            @mask_array[layer][row, column] = 1.0 / @dropouts_array[layer]
          else
            @mask_array[layer][row, column] = 0.0 / @dropouts_array[layer]
          end
          column += 1
        end
        row += 1
      end
      @a_array[layer] = @a_array.clone[layer].hadamard_product(@mask_array[layer])
      layer += 1
    end
  end

  def apply_cost(data_y)
    tmp = nil
    if @cost_function == 'mse'
      tmp = @cost.mse_cost(@a_array.last, data_y)
    elsif @cost_function == 'crossentropy'
      tmp = @cost.crossentropy_cost(@a_array.last, data_y)
    end
    tmp
  end

  def apply_activ(layer, activation)
    tmp = nil
    if activation == 'relu'
      tmp = @activation.relu(layer)
    elsif activation == 'leaky_relu'
      tmp = @activation.leaky_relu(layer)
    elsif activation == 'softmax'
      tmp = @activation.softmax(layer)
    end
    tmp
  end

  def apply_deriv(layer, activation)
    tmp = nil
    if activation == 'relu'
      tmp = @activation.relu_d(layer)
    elsif activation == 'leaky_relu'
      tmp = @activation.leaky_relu_d(layer)
    elsif activation == 'nil'
      tmp = layer
    end
    tmp
  end
end
