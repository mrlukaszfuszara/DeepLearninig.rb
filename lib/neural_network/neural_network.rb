require 'digest'

require './lib/util/matrix_math'
require './lib/util/generators'
require './lib/util/activations'
require './lib/util/costs'

class NeuralNetwork
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

        forward_propagation(x)

        stat = "Epoch: #{t}, of: #{epochs} epochs, iter: #{i}, of: #{train_data_x.size} iters in epoch, train error: #{apply_cost(y)}"

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
    serialized_array = Marshal.dump(@weights_array)
    File.open(path, 'wb') { |f| f.write(serialized_array) }
    File.open(path + '.sha512', 'w') { |f| f.write(Digest::SHA512.file(path)) }
  end

  def save_architecture(path)
    serialized_array = Marshal.dump([@layers_array, @activations_array, @dropouts_array, @optimizer, @cost_function, @learning_rate, @decay_rate, @momentum])
    File.open(path, 'wb') { |f| f.write(serialized_array) }
    File.open(path + '.sha512', 'w') { |f| f.write(Digest::SHA512.file(path)) }
  end

  def load_weights(key, path)
    tmp = nil
    if File.read(key) == Digest::SHA512.file(path).to_s
      tmp = Marshal.load File.open(path, 'rb')
    else
      puts 'SHA512 sum does not match'
    end

    @weights_array = tmp
  end

  def load_architecture(key, path)
    tmp = nil
    if File.read(key) == Digest::SHA512.file(path).to_s
      tmp = Marshal.load File.open(path, 'rb')
    else
      puts 'SHA512 sum does not match'
    end

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
    Matrix.build(@layers_array[counter], @layers_array[counter + 1]) { rand(0.0..0.01) * Math.sqrt(2.0 / (@layers_array[counter + 1] + @layers_array[counter])) }
  end

  def create_deltas
    @v_delta_w_array = []
    @s_delta_w_array = []
    layer = 0
    while layer < @layers_array.size - 1
      @v_delta_w_array << Matrix.build(@layers_array[layer], @layers_array[layer + 1]) { 10**-8 }
      @s_delta_w_array << Matrix.build(@layers_array[layer], @layers_array[layer + 1]) { 10**-8 }
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
        @a_array[layer] = @z_array[layer]
      else
        @z_array[layer] = @weights_array[layer - 1].transpose * @a_array[layer - 1]
        @a_array[layer] = apply_activ(@z_array[layer], @activations_array[layer])
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
    delta_z = []
    delta_x = []
    delta_w = []
    i = @layers_array.size - 1
    while i > 0
      if i == @layers_array.size - 1
        delta_z[i] = @a_array.last - data_y
        tmp = []
        j = 0
        while j < @a_array.last.row_size
          tmp[j] = []
          k = 0
          while k < @a_array.last.column_size
            if j == k
              tmp[j][k] = @a_array.last[j, k] * (1.0 - @a_array.last[j, k])
            else
              tmp[j][k] = -1.0 * @a_array[i][j, k]**2
            end
            k += 1
          end
          j += 1
        end
        delta_x[i] = @weights_array[i - 1] * delta_z[i].hadamard_product(Matrix[*tmp].elementwise_var_add(-1.0)).hadamard_product(@mask_array[i])
      else
        delta_z[i] = delta_x[i + 1].hadamard_product(apply_activ(@a_array[i], @activations_array[i]))
        delta_x[i] = @weights_array[i - 1] * delta_z[i].hadamard_product(@mask_array[i])
        delta_w[i] = delta_z[i] * @z_array[i + 1].transpose
      end
      i -= 1
    end
    delta_w
  end

  def mse_delta(data_y)
    delta_z = []
    delta_x = []
    delta_w = []
    i = @layers_array.size - 1
    while i > 0
      if i == @layers_array.size - 1
        delta_z[i] = @a_array.last - data_y
        delta_x[i] = @weights_array[i - 1] * delta_z[i].hadamard_product(@mask_array[i])
      else
        delta_z[i] = delta_x[i + 1].hadamard_product(apply_activ(@a_array[i], @activations_array[i]))
        delta_x[i] = @weights_array[i - 1] * delta_z[i].hadamard_product(@mask_array[i])
        delta_w[i] = delta_z[i] * @z_array[i + 1].transpose
      end
      i -= 1
    end
    delta_w
  end

  def update_weights
    if @optimizer == 'BGD'
      layer = @weights_array.size - 1
      while layer > 0
        @weights_array[layer] -= @learning_rate * @delta_w_array[layer]
        layer -= 1
      end
    elsif @optimizer == 'BGDwM'
      layer = @weights_array.size - 1
      while layer > 0
        @v_delta_w_array[layer] = @momentum[0] * @v_delta_w_array[layer] + (1.0 - @momentum[0]) * @delta_w_array[layer]
        @weights_array[layer] -= @learning_rate * @v_delta_w_array[layer]
        layer -= 1
      end
    elsif @optimizer == 'RMSprop'
      layer = @weights_array.size - 1
      while layer > 0
        @s_delta_w_array[layer] = @momentum[0] * @s_delta_w_array[layer] + (1.0 - @momentum[0]) * @delta_w_array[layer].pow(2)
        @weights_array[layer] -= @learning_rate * @delta_w_array[layer].elementwise_matrix_div(@s_delta_w_array[layer].sqrt.elementwise_var_add(10**-8))
        layer -= 1
      end
    elsif @optimizer == 'Adam'
      layer = @weights_array.size - 1
      while layer > 0
        @v_delta_w_array[layer] = @momentum[0] * @v_delta_w_array[layer] + (1.0 - @momentum[0]) * @delta_w_array[layer]
        @s_delta_w_array[layer] = @momentum[1] * @s_delta_w_array[layer] + (1.0 - @momentum[1]) * @delta_w_array[layer].pow(2)
        @v_delta_w_array[layer] = @v_delta_w_array[layer].elementwise_var_div(1.0 - @momentum[0])
        @s_delta_w_array[layer] = @s_delta_w_array[layer].elementwise_var_div(1.0 - @momentum[1])
        @weights_array[layer] -= @learning_rate * @v_delta_w_array[layer].elementwise_matrix_div(@s_delta_w_array[layer].sqrt.elementwise_var_add(@momentum[2]))
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
      @a_array[layer] = @a_array[layer].hadamard_product(@mask_array[layer])
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
