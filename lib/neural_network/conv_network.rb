class ConvNetwork
  def initialize
    @mm = MatrixMath.new
    @cm = ConvMath.new

    @g = Generators.new
    @a = Activations.new

    @array_of_layers = []
    @array_of_activations = []
    @array_of_channels = []
    @array_of_filters = []
    @array_of_paddings = []
    @array_of_strides = []
    @array_of_labels = []
    @array_of_a = []

    @array_of_weights = []
    @array_of_pool = []
  end

  def input(activation, channels, filter_size, padding, stride, label = nil)
    add_convnet(activation, channels, filter_size, padding, stride, label)
  end

  def add_convnet(activation, channels, filter_size, padding, stride, label = nil)
    @array_of_activations << activation
    @array_of_channels << channels
    @array_of_filters << filter_size
    @array_of_paddings << padding
    @array_of_strides << stride
    @array_of_labels << [label, (@array_of_channels.size - 1)]
    @array_of_pool << 0
  end

  def add_maxpool(filter_size, padding, stride, label = nil)
    @array_of_activations << 0
    @array_of_channels << 0
    @array_of_filters << filter_size
    @array_of_paddings << padding
    @array_of_strides << stride
    @array_of_labels << [label, (@array_of_channels.size - 1)]
    @array_of_pool << 'max'
  end

  def add_avgpool(filter_size, padding, stride, label = nil)
    @array_of_activations << 0
    @array_of_channels << 0
    @array_of_filters << filter_size
    @array_of_paddings << padding
    @array_of_strides << stride
    @array_of_labels << [label, (@array_of_channels.size - 1)]
    @array_of_pool << 'avg'
  end

  def return_flatten(label = nil)
    if !label.nil?
      lbl = 0
      while lbl < @array_of_channels.size
        tmp = @array_of_a[@array_of_labels[lbl][1]].flatten if lbl == label
        lbl += 1
      end
    else
      tmp = @array_of_a.last.flatten
    end
    tmp
  end

  def compile
    i = 0
    while i < @array_of_channels.size
      @array_of_weights << create_weights(i)
      i += 1
    end
  end

  def fit(path)
    pathes = generate_images_path(path)

    img_load = ImageLoader.new

    element = 0
    while element < pathes.size
      img = img_load.load_image(path + '\\' + pathes[element])

      layer = 0
      while layer < @array_of_channels.size
        if layer.zero?
          @array_of_a[layer] = @cm.conv2d(img, @array_of_weights[layer], @array_of_paddings[layer], @array_of_strides[layer])
        else
          if @array_of_pool[layer] == 0
            @array_of_a[layer] = @cm.conv2d(@array_of_a[layer - 1], @array_of_weights[layer], @array_of_paddings[layer], @array_of_strides[layer])
          elsif @array_of_pool[layer] == 'max'
            @array_of_a[layer] = @cm.max_pooling(@array_of_a[layer - 1], @array_of_filters[layer], @array_of_paddings[layer], @array_of_strides[layer])
          elsif @array_of_pool[layer] == 'avg'
            @array_of_a[layer] = @cm.average_pooling(@array_of_a[layer - 1], @array_of_filters[layer], @array_of_paddings[layer], @array_of_strides[layer])
          end
        end
        layer += 1
      end

      element += 1
    end
  end

  private

  def create_weights(i)
    @g.random_volume(@array_of_filters[i], @array_of_filters[i], @array_of_channels[i], -1..1)
  end

  def generate_images_path(dir_path)
    tmp = Dir.pwd
    Dir.chdir(dir_path)
    img = Dir.glob('*.png')
    Dir.chdir(tmp)
    img
  end
end
