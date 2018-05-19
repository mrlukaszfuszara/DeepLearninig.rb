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

    filter_center = filter[0].size / 2

    array = []
    i = 0
    while i < filter.size
      out = []
      j = 0
      while j < volume.size
        out[j] = []
        k = 0
        while k < volume[j].size
          out[j][k] = []
          l = 0
          while l < volume[j][0].size
            out[j][k][l] = 0
            m = 0
            while m < filter[i].size
              mm = filter[i].size - 1 - m
              n = 0
              while n < filter[i].size
                nn = filter[i].size - 1 - n
                kk = k + (m - filter_center)
                ll = l + (n - filter_center)
                if kk >= 0 && kk < volume[j].size - 1 && ll >= 0 && ll < volume[j][0].size - 1
                  out[j][k][l] += volume[j][kk][ll] * filter[i][mm][nn]
                end
                n += 1
              end
              m += 1
            end
            l += stride
          end
          out[j][k] = out[j][k] - [nil]
          k += stride
        end
        out[j] = out[j] - [nil]
        j += 1
      end
      array[i] = []
      j = 0
      while j < out[0].size
        array[i][j] = []
        k = 0
        while k < out[0][0].size
          array[i][j][k] = 0
          l = 0
          while l < out.size
            array[i][j][k] += out[l][j][k]
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