class Activations
  def initialize
    @mm = MatrixMath.new
  end

  def softmax(matrix)
    array = matrix.to_a
    i = 0
    while i < array.size
      tmp0 = array[i].max
      tmp1 = array[i].collect { |e| Math.exp(e - tmp0) }
      tmp2 = tmp1.inject(:+)
      array[i] = tmp1.collect { |e| e / tmp2 } 
      i += 1
    end
    Matrix[*array]
  end

  def relu(matrix)
    matrix.collect { |e| e > 0 ? e : 0.0 }
  end

  def relu_d(matrix)
    matrix.collect { |e| e > 0 ? 1.0 : 0.0 }
  end

  def leaky_relu(matrix)
    matrix.collect { |e| e * 0.01 > 0 ? e : e * 0.01 }
  end

  def leaky_relu_d(matrix)
    matrix.collect { |e| e > 0 ? 1.0 : 0.01 }
  end

  def relu_conv(volume)
    array = []
    i = 0
    while i < volume.size
      array[i] = []
      j = 0
      while j < volume[0].size
        array[i][j] = []
        k = 0
        while k < volume[i][j].size
          if volume[i][j][k] > 0.0
            array[i][j][k] = volume[i][j][k]
          else
            array[i][j][k] = 0.0
          end
          k += 1
        end
        j += 1
      end
      i += 1
    end
    array
  end

  def leaky_relu_conv(volume)
    array = []
    i = 0
    while i < volume.size
      array[i] = []
      j = 0
      while j < volume[0].size
        array[i][j] = []
        k = 0
        while k < volume[0][0].size
          tmp = 0.01 * volume[i][j][k]
          if volume[i][j][k] > tmp
            array[i][j][k] = volume[i][j][k]
          else
            array[i][j][k] = tmp
          end
          k += 1
        end
        j += 1
      end
      i += 1
    end
    array
  end

  private

  def matrix_check(variable)
    logic = nil
    if variable.class == Array
      if variable.all? { |e| e.class == Array }
        logic = 2
      else
        logic = 1
      end
    else
      logic = 0
    end
    logic
  end
end
