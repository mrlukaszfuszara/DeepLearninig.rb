require './lib/layers/dense_layer'

class Dense
  def initialize
    @array_of_classes = []
    @counter = 0
  end

  def add_dense(batch_size, activation)
    @last_size = batch_size if @counter.zero?
    @array_of_classes << DenseLayer.new(batch_size, activation, @last_size)
    @counter += 1
    @last_size = batch_size
  end

  def compile
    tmp = 0
    i = 0
    while i < @array_of_classes.size
      tmp = @array_of_classes[i].compile_data
      i += 1
    end
    tmp
  end

  def fit(data_x, data_y, epochs, alpha)
    data_x_chunked = chunk_data_x(data_x)
    n = 0
    while n < epochs
      m = 0
      while m < data_x_chunked.size
        i = 0
        while i < @array_of_classes.size
          if i.zero?
            @array_of_classes[i].fit_forward(data_x_chunked[m])
          else
            @array_of_classes[i].fit_forward(@array_of_classes[i - 1].output)
          end
          i += 1
        end
        i = @array_of_classes.size - 1
        while i > 1
          layer = false
          if i == @array_of_classes.size - 1
            layer = true
            @array_of_classes[i].fit_backward(layer, data_y, @array_of_classes[i - 1].output,
              @array_of_classes[i].weights)
          else
            layer = false
            @array_of_classes[i].fit_backward(layer, nil, @array_of_classes[i - 1].output,
              @array_of_classes[i + 1].weights, @array_of_classes[i + 1].delta)
          end
          i -= 1
        end
        i = @array_of_classes.size - 1
        while i > 1
          @array_of_classes[i].update_weights(@array_of_classes[i].delta, alpha)
          i -= 1
        end
        m += 1
      end
      puts (m * n).to_s + '/' + (epochs * data_x_chunked.size).to_s + ' error: ' + @array_of_classes.last.error.to_s
      n += 1
    end
    @array_of_classes.last.output
  end

  private

  def chunk_data_x(data_x)
    array = []
    i = 0
    while i < data_x.size
      array << data_x[i...(i + @array_of_classes[0].batch_size)]
      i += @array_of_classes[0].batch_size
    end
    if array[-1].size != array[-2].size
      array.pop
    end
    array[0].transpose
  end
end
