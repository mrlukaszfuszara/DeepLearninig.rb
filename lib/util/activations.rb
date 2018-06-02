class Activations
  def softmax(matrix)
    array = matrix.to_a.clone
    i = 0
    while i < array.size
      tmp2 = []
      j = 0
      while j < array[0].size
        tmp2[j] = Math.exp(array[i][j])
        j += 1
      end
      tmp3 = tmp2.inject(:+)
      j = 0
      while j < array[0].size
        tmp2[j] = tmp2[j] / tmp3
        j += 1
      end
      array[i] = tmp2
      i += 1
    end
    Matrix[*array]
  end

  def tanh(matrix)
    matrix.map { |e| (Math.exp(e) - Math.exp(-e)) / Math.exp(e) + Math.exp(-e) }
  end

  def tanh_d(matrix)
    matrix.map { |e| 1.0 - Math.tanh(e**2) }
  end

  def sigmoid(matrix)
    matrix.map { |e| 1.0 / (1.0 + Math.exp(-e)) }
  end

  def sigmoid_d(matrix)
    matrix.map { |e| e * (1.0 - e) }
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
