class ConvMath
  def conv2d(matrix, filter = nil, padding = 1, stride = 1)
    siz = ((matrix.size + (2.0 * padding) - filter[0].size) / stride.to_f) + 1

    start_size = matrix[0][0].size

    i = 0
    while i < padding
      j = 0
      while j < matrix.size
        matrix[j].push([0.0] * start_size)
        matrix[j].unshift([0.0] * start_size)
        j += 1
      end
      matrix.push(Array.new(matrix.size + 2, [0.0] * start_size))
      matrix.unshift(Array.new(matrix.size + 1, [0.0] * start_size))
      i += 1
    end

    array = []
    tmp = []
    i = 0
    while i < siz
      array[i] = []
      tmp[i] = []
      j = 0
      while j < siz
        tmp[i][j] = 0
        ch = 0
        while ch < matrix[0][0].size
          k = 0
          while k < filter.size
            l = 0
            while l < filter[0].size
              tmp[i][j] += (matrix[i + k][j + l][ch] * filter[k][l][ch]).floor
              l += 1
            end
            k += 1
          end
          ch += 1
        end
        array[i][j] = tmp[i]
        j += stride
      end
      i += stride
    end
    array
  end

  def max_pooling(matrix, pooling_size = 3, padding = 0, stride = 1)
    siz = ((matrix.size + (2.0 * padding) - pooling_size) / stride.to_f) + 1

    i = 0
    while i < matrix.size
      j = 0
      while j < matrix[i].size
        tmp = VectorizeArray.new
        matrix[i][j] = tmp.var_only(matrix[i][j])
        j += 1
      end
      i += 1
    end

    start_size = matrix[0][0].size

    i = 0
    while i < padding
      j = 0
      while j < matrix.size
        matrix[j].push([0.0] * start_size)
        matrix[j].unshift([0.0] * start_size)
        j += 1
      end
      matrix.push(Array.new(matrix.size + 2, [0.0] * start_size))
      matrix.unshift(Array.new(matrix.size + 1, [0.0] * start_size))
      i += 1
    end

    array = []
    tmp = []
    i = 0
    while i < siz
      array[i] = []
      tmp[i] = []
      j = 0
      while j < siz
        tmp[i][j] = []
        ch = 0
        while ch < matrix[0][0].size
          tmp[i][j][ch] = []
          k = 0
          while k < filter.size
            l = 0
            while l < filter[0].size
              tmp[i][j][k] << matrix[i + k][j + l][ch]
              l += 1
            end
            k += 1
          end
          array[i][j][ch] = tmp[i][j][ch].flatten.max
          ch += 1
        end
        j += stride
      end
      i += stride
    end
    array
  end

  def average_pooling(matrix, pooling_size = 3, padding = 0, stride = 1)
    siz = ((matrix.size + (2.0 * padding) - pooling_size) / stride.to_f) + 1

    i = 0
    while i < matrix.size
      j = 0
      while j < matrix[i].size
        tmp = VectorizeArray.new
        matrix[i][j] = tmp.var_only(matrix[i][j])
        j += 1
      end
      i += 1
    end

    start_size = matrix[0][0].size

    i = 0
    while i < padding
      j = 0
      while j < matrix.size
        matrix[j].push([0.0] * start_size)
        matrix[j].unshift([0.0] * start_size)
        j += 1
      end
      matrix.push(Array.new(matrix.size + 2, [0.0] * start_size))
      matrix.unshift(Array.new(matrix.size + 1, [0.0] * start_size))
      i += 1
    end

    array = []
    tmp = []
    i = 0
    while i < siz
      array[i] = []
      tmp[i] = []
      j = 0
      while j < siz
        tmp[i][j] = []
        ch = 0
        while ch < matrix[0][0].size
          tmp[i][j][ch] = []
          k = 0
          while k < filter.size
            l = 0
            while l < filter[0].size
              tmp[i][j][k] << matrix[i + k][j + l][ch]
              l += 1
            end
            k += 1
          end
          array[i][j][ch] = tmp[i][j][ch].flatten.inject(:+) / (pooling_size**2).to_f
          ch += 1
        end
        j += stride
      end
      i += stride
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