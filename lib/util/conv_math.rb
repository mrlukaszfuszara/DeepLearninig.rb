class ConvMath
  def conv2d(matrix, filter, padding = 1, stride = 1)
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

    tmp = []
    i = 0
    while i < siz
      tmp[i] = []
      j = 0
      while j < siz
        tmp[i][j] = []
        m = 0
        while m < filter[0][0].size
          tmp[i][j][m] = 0
          k = 0
          while k < filter.size
            l = 0
            while l < filter[0].size
              ch = 0
              while ch < matrix[0][0].size
                tmp[i][j][m] += (matrix[i + k][j + l][ch] * filter[k][l][m]).floor
                ch += 1
              end
              l += 1
            end
            k += 1
          end
          m += 1
        end
        tmp[i][j] = tmp[i][j]
        j += stride
      end
      tmp[i] = tmp[i] - [nil]
      i += stride
    end
    tmp - [nil]
  end

  def max_pooling(matrix, filter = 3, padding = 0, stride = 1)
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

    tmp = []
    i = 0
    while i < siz
      tmp[i] = []
      j = 0
      while j < siz
        tmp[i][j] = []
        k = 0
        while k < filter.size
          l = 0
          while l < filter.size
            ch = 0
            while ch < matrix[0][0].size
              tmp[i][j] << matrix[i + k][j + l][ch]
              ch += 1
            end
            l += 1
          end
          k += 1
        end
        tmp[i][j] = tmp[i][j].flatten.max
        j += stride
      end
      i += stride
    end
    tmp - [nil]
  end

  def average_pooling(matrix, pooling_size = 3, padding = 0, stride = 1)
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

    tmp = []
    i = 0
    while i < siz
      tmp[i] = []
      j = 0
      while j < siz
        tmp[i][j] = []
        k = 0
        while k < filter.size
          l = 0
          while l < filter.size
            ch = 0
            while ch < matrix[0][0].size
              tmp[i][j] << matrix[i + k][j + l][ch]
              ch += 1
            end
            l += 1
          end
          k += 1
        end
        tmp[i][j] = (tmp[i][j].flatten.inject(:+) / tmp[i][j].flatten.size.to_f).floor
        j += stride
      end
      i += stride
    end
    tmp - [nil]
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