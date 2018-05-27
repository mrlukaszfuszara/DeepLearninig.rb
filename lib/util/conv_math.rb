class ConvMath
  def padding(volume, pad)
    pa = 0
    while pa < pad
      i = 0
      while i < volume.size
        volume[i].push([0.0] * volume[i].size)
        volume[i].unshift([0.0] * (volume[i].size - 1))
        j = 0
        while j < volume[i].size
          volume[i][j].push(0.0)
          j += 1
        end
        j = 0
        while j < volume[i].size
          volume[i][j].unshift(0.0)
          j += 1
        end
        i += 1
      end
      pa += 1
    end
    volume
  end

  def splice_with_stride(volume, chunk, stride)
    array = []
    channel = 0
    while channel < volume.size
      array[channel] = []
      row = chunk
      while row < volume[0].size
        array[channel][row] = []
        column = chunk
        while column < volume[0].size
          array[channel][row][column] = []
          f0 = row - chunk
          while f0 < row
            array[channel][row][column][f0] = []
            f1 = column - chunk
            while f1 < column
              array[channel][row][column][f0][f1] = volume[channel][f0][f1]
              f1 += 1
            end
            array[channel][row][column][f0] -= [nil]
            f0 += 1
          end
          array[channel][row][column] -= [nil]
          column += stride
        end
        array[channel][row] -= [nil]
        row += stride
      end
      array[channel] -= [nil]
      channel += 1
    end
    array
  end

  def sum_channels(volume)
    array = []
    i = 0
    while i < volume.size
      array[i] = []
      j = 0
      while j < volume[0][0].size
        array[i][j] = []
        k = 0
        while k < volume[0][0].size
          array[i][j][k] = 0.0
          l = 0
          while l < volume[0].size
            array[i][j][k] += volume[i][l][j][k]
            l += 1
          end
          k += 1
        end
        j += 1
      end
      i += 1
    end
    array
  end

  def conv2d(volume, filter)
    array = []
    i = 0
    while i < volume.size
      array[i] = []
      j = 0
      while j < volume[0].size
        array[i][j] = []
        k = 0
        while k < volume[0].size
          array[i][j][k] = Matrix[*volume[i][j][k]].hadamard_product(Matrix[*filter]).to_a.flatten.inject(:+)
          k += 1
        end
        j += 1
      end
      i += 1
    end
    array
  end

  def max_pooling(volume)
    array = []
    i = 0
    while i < volume.size
      array[i] = []
      j = 0
      while j < volume[0].size
        array[i][j] = []
        k = 0
        while k < volume[0].size
          array[i][j][k] = volume[i][j][k].flatten.max
          k += 1
        end
        j += 1
      end
      i += 1
    end
    array
  end
end
