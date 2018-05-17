class ConvNetwork
  def initialize
    @mm = MatrixMath.new
    @cm = ConvMath.new

    @g = Generators.new
    @a = Activations.new

    @array_of_activations = []
    @array_of_pool = []
    @array_of_channels = []
    @array_of_filters = []
    @array_of_paddings = []
    @array_of_strides = []

    @array_of_weights = []

    @array_of_elements = []
  end

  def input(activation, channels, filter_size, padding, stride)
    add_convnet(activation, channels, filter_size, padding, stride)
  end

  def add_convnet(activation, channels, filter_size, padding, stride)
    @array_of_activations << activation
    @array_of_channels << channels
    @array_of_filters << filter_size
    @array_of_paddings << padding
    @array_of_strides << stride
    @array_of_pool << 0
  end

  def add_maxpool(filter_size, padding, stride)
    @array_of_activations << 0
    @array_of_channels << 0
    @array_of_filters << filter_size
    @array_of_paddings << padding
    @array_of_strides << stride
    @array_of_pool << 'max'
  end

  def compile
    i = 0
    while i < @array_of_channels.size
      @array_of_weights << create_weights(i)
      i += 1
    end
  end

  def fit(path_to_files, files)
    img_load = ImageLoader.new

    element = 0
    while element < files.size
      @array_of_z = []
      @array_of_a = []

      img = img_load.load_image(path_to_files + '\\' + files[element])

      layer = 0
      while layer < @array_of_channels.size
        if layer.zero?
          @array_of_z[layer] = @cm.conv2d(img, @array_of_weights[layer], @array_of_paddings[layer], @array_of_strides[layer])
          @array_of_a[layer] = apply_activ(@array_of_z[layer], @array_of_activations[layer])
        else
          if @array_of_pool[layer] == 0
            @array_of_z[layer] = @cm.conv2d(@array_of_a[layer - 1], @array_of_weights[layer], @array_of_paddings[layer], @array_of_strides[layer])
            @array_of_a[layer] = apply_activ(@array_of_z[layer], @array_of_activations[layer])
          elsif @array_of_pool[layer] == 'max'
            @array_of_a[layer] = @cm.max_pooling(@array_of_a[layer - 1], @array_of_filters[layer], @array_of_paddings[layer], @array_of_strides[layer])
          end
        end
        layer += 1
      end
      @array_of_elements << @array_of_a

      element += 1

      str = 'Image: ' + element.to_s + ', of: ' + files.size.to_s

      puts str

      windows_size = IO.console.winsize[1].to_f - 20.0

      max_val = files.size.to_f
      current_val = element.to_f
      pg_bar = current_val / max_val

      puts '[' + '#' * (pg_bar * windows_size).floor + '*' * (windows_size - (pg_bar * windows_size)).floor + '] ' + (100 * pg_bar).floor.to_s + '%'
    end
  end

  def return_flatten
    tmp = []
    element = 0
    while element < @array_of_elements.size
      tmp << @array_of_elements[element].flatten
      element += 1
    end
    tmp
  end

  def save_weights(path)
    serialized_array = Marshal.dump([@array_of_weights])
    File.open(path, 'wb') { |f| f.write(serialized_array) }
  end

  def save_architecture(path)
    serialized_array = Marshal.dump([@array_of_activations, @array_of_pool, @array_of_channels, @array_of_filters, @array_of_paddings, @array_of_strides])
    File.open(path, 'wb') { |f| f.write(serialized_array) }
  end

  def load_weights(path)
    tmp = Marshal.load File.open(path, 'rb')

    @array_of_weights = tmp[0]
  end

  def load_architecture(path)
    tmp = Marshal.load File.open(path, 'rb')

    @array_of_channels = tmp[2]
    layers = @array_of_channels.size
    @array_of_channels = []

    @array_of_activations = tmp[0]
    @array_of_pool = tmp[1]
    @array_of_filters = tmp[3]
    @array_of_paddings = tmp[4]
    @array_of_strides = tmp[5]
    
    i = 0
    while i < layers
      add_convnet(@array_of_activations[layer], @array_of_channels[layer], @array_of_filters[layer], @array_of_paddings[layer], @array_of_strides[layer])
      i += 1
    end
  end

  private

  def create_weights(i)
    @g.random_volume(@array_of_channels[i], @array_of_filters[i], -1..1)
  end

  def apply_activ(layer, activation)
    if activation == 'nil'
      tmp = layer
    elsif activation == 'relu'
      tmp = @a.relu_conv(layer)
    elsif activation == 'leaky_relu'
      tmp = @a.leaky_relu_conv(layer)
    end
    tmp
  end
end
