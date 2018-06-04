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

  def conv2d(volume, filter, stride)
    filter_size = filter[0].size
    filter = fft_matrix(filter)
    splice_with_stride(volume, filter_size, stride) { |e| (ifft_matrix((Matrix[*fft_matrix(e)].hadamard_product(Matrix[*filter])).to_a)).flatten.inject(:+) }
  end

  def max_pooling(volume, filter, stride)
    filter_size = filter
    splice_with_stride(volume, filter_size, stride) { |e| e.to_a.flatten.max }
  end

  def avg_pooling(volume, filter, stride)
    filter_size = filter
    splice_with_stride(volume, filter_size, stride) { |e| e.to_a.flatten.inject(:+) / e.size }
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

  private

  def fft_matrix(matrix)
    array = []
    i = 0
    while i < matrix.size
      array[i] = fft(matrix[i])
      i += 1
    end
    array
  end

  def ifft_matrix(matrix)
    array = []
    i = 0
    while i < matrix.size
      array[i] = ifft(matrix[i])
      i += 1
    end
    array
  end

  def fft(array)
    n = array.size
    return array if n <= 1

    even_vals = array.values_at(* array.each_index.select {|i| i.even?})
    odd_vals = array.values_at(* array.each_index.select {|i| i.odd?})

    fft(even_vals)
    fft(odd_vals)

    k = 0
    while k < n / 2
      t = Complex.polar(1.0, -2.0 * Math::PI * k / n) * odd_vals[k]
      array[k] = even_vals[k] + t
      array[k + n / 2] = even_vals[k] - t
      k += 1
    end
    array
  end

  def ifft(array)
    tmp = array.map { |e| e.conj }
    tmp = fft(tmp)
    tmp = tmp.map { |e| e.conj }
    tmp = tmp.map { |e| e / array.size }
    tmp.map { |e| e.abs }
  end

  def splice_with_stride(volume, chunk, stride)
    array = []
    out = []
    channel = 0
    while channel < volume.size
      array[channel] = []
      out[channel] = []
      row = chunk
      while row < volume[0].size
        array[channel][row] = []
        out[channel][row] = []
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
          out[channel][row][column] = yield array[channel][row][column]
          column += stride
        end
        array[channel][row] -= [nil]
        out[channel][row] -= [nil]
        row += stride
      end
      array[channel] -= [nil]
      out[channel] -= [nil]
      channel += 1
    end
    out
  end
end
