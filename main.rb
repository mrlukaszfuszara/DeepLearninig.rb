require './lib/math'
require './lib/layers/dense'
require './lib/layers/dropout'
require './lib/layers/flatten'

class RuNNet
  def initialize
    @array_of_classes = Array.new
    @last_size = 0
  end

  def add_dense(activation, size)
    @array_of_classes << Dense.new(activation, size, @last_size)
    @last_size = size
  end

  def add_dropout(rate)
    @array_of_classes << Dropout.new(rate)
  end

  def add_flatten
    @array_of_classes << Flatten.new
  end

  def input(data_x, data_y, batch_size, epochs)
    @data_x = data_x
    @data_y = data_y
    @batch_size = batch_size
    @epochs = epochs
  end

  def compile
    tmp = 0
    n = 0
    for i in @array_of_classes
      if n == 0
        tmp = i.compile_data(@data_x)
      else
        tmp = i.compile_data(tmp)
      end
      n +=1 
    end
    tmp
  end

  def fit
    tmp = Array.new
    n = 0
    while n < @epochs
      w = Array.new
      d = Array.new
      o = Array.new
      i = 0
      while i < @array_of_classes.size
        if i == 0
          tmp[i] = @array_of_classes[i].fit_forward(@data_x)
        else
          tmp[i] = @array_of_classes[i].fit_forward(o[i - 1])
        end
        o[i] = tmp[i]
        i += 1
      end
      i = @array_of_classes.size - 1
      while i > 0
        if i == @array_of_classes.size - 1
          last_layer = true
          tmp_a = @array_of_classes[i].fit_backward(0, 0, 0, @data_y, last_layer)
        else
          last_layer = false
          tmp_a = @array_of_classes[i].fit_backward(w[i + 1], d[i + 1], o[i], 0, last_layer)
        end
        w[i] = tmp_a[0]
        d[i] = tmp_a[1]
        tmp[i] = tmp_a[2]
        i -= 1
      end
      n += 1
    end
    tmp.last
  end
end

input = [[0.9, 0.5, 0.9], [0.9, 0.5, 0.9], [0.9, 0.5, 0.9]]
a = RuNNet.new
a.add_dense('sigmoid', 3)
#a.add_dropout(0.05)
a.add_dense('sigmoid', 4)
a.add_dense('tanh', 3)
#a.add_flatten
a.input(input, [0.9, 0.5, 0.9], 25, 100)
a.compile
p a.fit