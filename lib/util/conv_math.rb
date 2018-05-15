class ConvMath
  def conv2d(volume, filter, padding, stride)
    chan_size = volume[0][0].size

    height = volume.size
    width = volume[0].size

    margin = [0.0] * chan_size

    i = 0
    while i < padding
      j = 0
      while j < height
        volume[j].push(margin)
        volume[j].unshift(margin)
        j += 1
      end
      volume.push(Array.new(volume.size + 2, margin))
      volume.unshift(Array.new(volume.size + 1, margin))
      i += 1
    end

    output_size = volume.size - filter[0].size + 1

    array = []
    i = 0
    while i < output_size
      array[i] = []
      j = 0
      while j < output_size
        array[i][j] = []
        k = 0
        while k < filter.size
          array[i][j][k] = 0
          l = 0
          while l < volume[0][0].size
            m = 0
            while m < filter[0].size
              array[i][j][k] += volume[i][j][l] * filter[k][m]
              m += 1
            end
            l += 1
          end
          k += 1
        end
        array[i][j] = array[i][j] - [nil]
        j += stride
      end
      array[i] = array[i] - [nil]
      i += stride
    end
    array - [nil]
  end

  def max_pooling(volume, filter_size, padding, stride)
    chan_size = volume[0][0].size

    height = volume.size
    width = volume[0].size

    margin = [0.0] * chan_size

    i = 0
    while i < padding
      j = 0
      while j < height
        volume[j].push(margin)
        volume[j].unshift(margin)
        j += 1
      end
      volume.push(Array.new(volume.size + 2, margin))
      volume.unshift(Array.new(volume.size + 1, margin))
      i += 1
    end

    output_size = volume.size - filter_size.size + 1

    array = []
    i = 0
    while i < output_size
      array[i] = []
      j = 0
      while j < output_size
        array[i][j] = []
        k = 0
        while k < volume[0][0].size
          array[i][j][k] = []
          l = 0
          while l < filter_size
            m = 0
            while m < filter_size
              array[i][j][k] << volume[i + l][j + m][k]
              m += 1
            end
            l += 1
          end
          array[i][j][k] = (array[i][j][k] - [nil]).max
          k += 1
        end
        array[i][j] = array[i][j] - [nil]
        j += stride
      end
      array[i] = array[i] - [nil]
      i += stride
    end
    array - [nil]
  end
end