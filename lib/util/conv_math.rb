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

    array = []
    chf = 0
    while chf < filter.size
      array[chf] = []
      chv = 0
      while chv < volume.size
        array[chf][chv] = []
        row = filter[0].size
        while row < volume[0].size
          array[chf][chv][row] = []
          column = filter[0].size
          while column < volume[0].size
            array[chf][chv][row][column] = []
            f0 = row - filter[0].size
            while f0 < row
              array[chf][chv][row][column][f0] = []
              f1 = column - filter[0].size
              while f1 < column
                array[chf][chv][row][column][f0] << volume[chv][f0][f1]
                f1 += 1
              end
              array[chf][chv][row][column][f0] -= [nil]
              f0 += 1
            end
            array[chf][chv][row][column] -= [nil]
            array[chf][chv][row][column] = (Matrix[*array[chf][chv][row][column]].hadamard_product(Matrix[*filter[chf]])).to_a
            column += stride
          end
          array[chf][chv][row] -= [nil]
          row += stride
        end
        array[chf][chv] -= [nil]
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
            out[i][j][k] << array[i][l][j][k]
            l += 1
          end
          out[i][j][k] = out[i][j][k].flatten.inject(:+)
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

    array = []
    chv = 0
    while chv < volume.size
      array[chv] = []
      row = filter_size
      while row < volume[0].size
        array[chv][row] = []
        column = filter_size
        while column < volume[0].size
          array[chv][row][column] = []
          f0 = row - filter_size
          while f0 < row
            f1 = column - filter_size
            while f1 < column
              array[chv][row][column] << volume[chv][f0][f1]
              f1 += 1
            end
            f0 += 1
          end
          array[chv][row][column] -= [nil]
          array[chv][row][column] = array[chv][row][column].flatten.max
          column += stride
        end
        array[chv][row] -= [nil]
        row += stride
      end
      array[chv] -= [nil]
      chv += 1
    end
    array
  end
end