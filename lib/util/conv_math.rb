class ConvMath
  def conv2d(matrix, filter, padding = 1, stride = 1)
    chan_size = matrix[0][0].size

    height = matrix.size
    width = matrix[0].size

    margin = [0.0] * chan_size

    i = 0
    while i < padding
      j = 0
      while j < height
        matrix[j].push(margin)
        matrix[j].unshift(margin)
        j += 1
      end
      matrix.push(Array.new(matrix.size + 2, margin))
      matrix.unshift(Array.new(matrix.size + 1, margin))
      i += 1
    end

    output_size = matrix.size - filter.size + 1

    array = []
    tmp = []
    i = 0
    while i < output_size
      array[i] = []
      tmp[i] = []
      j = 0
      while j < output_size
        array[i][j] = []
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
        array[i][j] = tmp[i][j] - [nil]
        j += stride
      end
      array[i] = array[i] - [nil]
      i += stride
    end
    array - [nil]
  end

  def max_pooling(matrix, filter_size = 3, padding = 0, stride = 1)
    chan_size = matrix[0][0].size

    height = matrix.size
    width = matrix[0].size

    margin = [0.0] * chan_size

    i = 0
    while i < padding
      j = 0
      while j < height
        matrix[j].push(margin)
        matrix[j].unshift(margin)
        j += 1
      end
      matrix.push(Array.new(matrix.size + 2, margin))
      matrix.unshift(Array.new(matrix.size + 1, margin))
      i += 1
    end

    output_size = matrix.size - filter_size.size + 1

    array = []
    tmp = []
    i = 0
    while i < output_size
      array[i] = []
      tmp[i] = []
      j = 0
      while j < output_size
        tmp[i][j] = []
        ch = 0
        while ch < matrix[0][0].size
          tmp[i][j][ch] = []
          k = 0
          while k < filter_size
            l = 0
            while l < filter_size
              tmp[i][j][ch] << (matrix[i + k][j + l][ch]).floor
              l += 1
            end
            k += 1
          end
          ch += 1
        end
        array[i] << tmp[i][j] - [nil]
        j += stride
      end
      i += stride
    end
    array = array - [nil]
    i = 0
    while i < array.size
      j = 0
      while j < array[i].size
        k = 0
        while k < array[i][j].size
          array[i][j][k] = array[i][j][k].max
          k += 1
        end
        j += 1
      end
      i += 1
    end
    array
  end
end