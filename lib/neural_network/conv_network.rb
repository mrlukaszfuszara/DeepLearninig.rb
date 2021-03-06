require './lib/neural_network/network'

require './lib/util/matrix_math'
require './lib/util/conv_math'
require './lib/util/generators'
require './lib/util/activations'

require './lib/util/image_loader'

class ConvNetwork < Network
  def initialize
    @convmath = ConvMath.new

    @activation = Activations.new

    @activations_array = []
    @pool_array = []
    @channels_array = []
    @filters_array = []
    @paddings_array = []
    @strides_array = []

    @weights_array = []

    @elements_array = []
  end

  def input(activation = 'nil', channels = 3, filter_size = 0, padding = 0, stride = 0)
    add_convnet(activation, channels, filter_size, padding, stride)
  end

  def add_convnet(activation, channels, filter_size, padding, stride)
    @activations_array << activation
    @channels_array << channels
    @filters_array << filter_size
    @paddings_array << padding
    @strides_array << stride
    @pool_array << 0
  end

  def add_maxpool(filter_size, padding, stride)
    @activations_array << nil
    @channels_array << 0
    @filters_array << filter_size
    @paddings_array << padding
    @strides_array << stride
    @pool_array << 1
  end

  def add_avgpool(filter_size, padding, stride)
    @activations_array << nil
    @channels_array << 0
    @filters_array << filter_size
    @paddings_array << padding
    @strides_array << stride
    @pool_array << 2
  end

  def compile
    i = 0
    while i < @channels_array.size
      @weights_array << create_weights(i)
      i += 1
    end
  end

  def fit(path_to_files, files)
    puts 'Lets start!'

    img_load = ImageLoader.new

    element = 0
    while element < files.size
      @z_array = []
      @a_array = []

      img = img_load.load_image(path_to_files + '\\' + files[element])

      forward_propagation(img)

      @elements_array << @a_array.last

      element += 1

      windows_size = IO.console.winsize[1].to_f - 20.0

      str = "Image: #{element}, of: #{files.size} images"

      max_val = files.size.to_f
      current_val = element.to_f
      pg_bar = current_val / max_val

      puts str
      puts '[' + '#' * (pg_bar * windows_size).floor + '*' * (windows_size - (pg_bar * windows_size)).floor + '] ' + (100 * pg_bar).floor.to_s + '%'
    end
  end

  def predict(path_to_files, files)
    img_load = ImageLoader.new

    element = 0
    while element < files.size
      @z_array = []
      @a_array = []

      img = img_load.load_image(path_to_files + '\\' + files[element])

      forward_propagation(img)

      @elements_array << @a_array.last
    end
  end

  def return_flatten
    tmp = []
    element = 0
    while element < @elements_array.size
      tmp << @elements_array[element].flatten
      element += 1
    end
    tmp
  end

  def save_weights(path)
    save_data(path, @weights_array)
  end

  def save_architecture(path)
    save_data(path, [@activations_array, @pool_array, @channels_array, @filters_array, @paddings_array, @strides_array])
  end

  def load_weights(key, path)
    tmp = load_data(key, path)
    @weights_array = tmp
  end

  def load_architecture(key, path)
    tmp = load_data(key, path)

    @channels_array = tmp[2]
    layers = @channels_array.size
    @channels_array = []

    @activations_array = tmp[0]
    @pool_array = tmp[1]
    @filters_array = tmp[3]
    @paddings_array = tmp[4]
    @strides_array = tmp[5]

    i = 0
    while i < layers
      add_convnet(@activations_array[layer], @channels_array[layer], @filters_array[layer], @paddings_array[layer], @strides_array[layer])
      i += 1
    end
  end

  private

  def create_weights(i)
    array = []
    ch = 0
    while ch < @channels_array[i]
      array[ch] = []
      j = 0
      while j < @filters_array[i]
        array[ch][j] = []
        k = 0
        while k < @filters_array[i]
          array[ch][j][k] = rand(-1..1)
          k += 1
        end
        j += 1
      end
      ch += 1
    end
    array
  end

  def forward_propagation(img)
    layer = 0
    while layer < @channels_array.size
      if layer.zero?
        @z_array[layer] = img
        @a_array[layer] = @z_array[layer]
      else
        if @pool_array[layer].zero?
          tmp = @convmath.padding(@a_array[layer - 1], @paddings_array[layer])
          tmp1 = []
          i = 0
          while i < @channels_array[layer]
            tmp1[i] = @convmath.conv2d(tmp, @weights_array[layer][i], @strides_array[layer])
            i += 1
          end
          @z_array[layer] = @convmath.sum_channels(tmp1)
          @a_array[layer] = apply_activ(@z_array[layer], @activations_array[layer])
        elsif @pool_array[layer] == 1
          tmp = @convmath.padding(@a_array[layer - 1], @paddings_array[layer])
          @z_array[layer] = @convmath.max_pooling(tmp, @filters_array[layer], @strides_array[layer])
          @a_array[layer] = @z_array[layer]
        elsif @pool_array[layer] == 2
          tmp = @convmath.padding(@a_array[layer - 1], @paddings_array[layer])
          @z_array[layer] = @convmath.avg_pooling(tmp, @filters_array[layer], @strides_array[layer])
          @a_array[layer] = @z_array[layer]
        end
      end
      p @a_array[layer]
      layer += 1
    end
  end

  def apply_activ(layer, activation)
    tmp = nil
    if activation == 'relu'
      tmp = @activation.relu_conv(layer)
    elsif activation == 'leaky_relu'
      tmp = @activation.leaky_relu_conv(layer)
    end
    tmp
  end

  def apply_deriv(layer, activation)
    tmp = nil
    if activation == 'relu'
      tmp = @activation.relu_conv_d(layer)
    elsif activation == 'leaky_relu'
      tmp = @activation.leaky_relu_conv_d(layer)
    end
    tmp
  end
end
