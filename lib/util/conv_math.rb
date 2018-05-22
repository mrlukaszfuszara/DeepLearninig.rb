class ConvMath
  def conv2d(volume, filter, padding, stride)
    pad = 0
    while pad < padding
      i = 0
      while i < volume.size
        volume[i].push([0.0] * volume[i].size)
        volume[i].unshift([0.0] * volume[i].size)
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
      pad += 1
    end

    output_size = volume[0].size - filter[0].size + 1

    array = []
    chf = 0
    while chf < filter.size
      array[chf] = []
      chv = 0
      while chv < volume.size
        array[chf][chv] = []
        row = filter[0].size
        while row < output_size
          array[chf][chv][row] = []
          column = filter[0].size
          while column < output_size
            array[chf][chv][row][column] = []
            f0 = row - filter[0].size
            while f0 < row
              array[chf][chv][row][column][f0] = []
              f1 = column - filter[0].size
              while f1 < column
                array[chf][chv][row][column][f0] << volume[chv][f0][f1]
                f1 += 1
              end
              f0 += 1
            end
            array[chf][chv][row][column].compact!
            array[chf][chv][row][column] = (Matrix[*array[chf][chv][row][column]].hadamard_product(Matrix[*filter[chf]])).to_a
            column += stride
          end
          array[chf][chv][row].compact!
          row += stride
        end
        array[chf][chv].compact!
        chv += 1
      end
      chf += 1
    end
    out = []
    i = 0
    while i < array.size
      out[i] = []
      j = 0
      while j < array[0][0].size
        out[i][j] = []
        k = 0
        while k < array[0][0].size
          out[i][j][k] = []
          l = 0
          while l < array[0].size
            out[i][j][k][l] = array[i][l][j][k].flatten.inject(:+)
            l += 1
          end
          out[i][j][k] = out[i][j][k].inject(:+)
          k += 1
        end
        j += 1
      end
      i += 1
    end
    out
  end

  def max_pooling(volume, filter_size, padding, stride)
    pad = 0
    while pad < padding
      i = 0
      while i < volume.size
        volume[i].push([0.0] * volume[i].size)
        volume[i].unshift([0.0] * volume[i].size)
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
      pad += 1
    end

    output_size = volume.size - filter_size.size + 1

    array = []
    chv = 0
    while chv < volume.size
      array[chv] = []
      row = filter_size
      while row < output_size
        array[chv][row] = []
        column = filter_size
        while column < output_size
          array[chv][row][column] = []
          f0 = row - filter_size
          while f0 < row
            array[chv][row][column][f0] = []
            f1 = column - filter_size
            while f1 < column
              array[chv][row][column][f0] << volume[chv][f0][f1]
              f1 += 1
            end
            f0 += 1
          end
          array[chv][row][column] = array[chv][row][column].flatten.compact
          array[chv][row][column] = array[chv][row][column].max
          column += stride
        end
        array[chv][row].compact!
        row += stride
      end
      array[chv].compact!
      chv += 1
    end
    array
  end
end