class Activations
  def softmax(matrix)
    array = matrix.to_a
    i = 0
    while i < array.size
      tmp0 = array[i].max
      tmp1 = array[i].map { |e| Math.exp(e - tmp0) }
      tmp2 = tmp1.inject(:+)
      array[i] = tmp1.map { |e| e / tmp2 }
      i += 1
    end
    Matrix[*array]
  end

  def relu(matrix)
    matrix.map { |e| e > 0 ? e : 0.0 }
  end

  def relu_d(matrix)
    matrix.map { |e| e > 0 ? 1.0 : 0.0 }
  end

  def leaky_relu(matrix)
    matrix.map { |e| e > 0 ? e : e * 0.01 }
  end

  def leaky_relu_d(matrix)
    matrix.map { |e| e > 0 ? 1.0 : 0.01 }
  end

  def relu_conv(volume)
    array = []
    i = 0
    while i < volume.size
      array[i] = volume[i].map { |e| e.map { |f| f > 0 ? f : 0.0 } }
      i += 1
    end
    array
  end

  def leaky_relu_conv(volume)
    array = []
    i = 0
    while i < volume.size
      array[i] = volume[i].map { |e| e.map { |f| f * 0.01 > 0 ? f : f * 0.01 } }
      i += 1
    end
    array
  end

  def relu_conv_d(volume)
    array = []
    i = 0
    while i < volume.size
      array[i] = volume[i].map { |e| e.map { |f| f > 0 ? 1.0 : 0.0 } }
      i += 1
    end
    array
  end

  def leaky_relu_conv_d(volume)
    array = []
    i = 0
    while i < volume.size
      array[i] = volume[i].map { |e| e.map { |f| f * 0.01 > 0 ? 1.0 : 0.01 } }
      i += 1
    end
    array
  end
end
