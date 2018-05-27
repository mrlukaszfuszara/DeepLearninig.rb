require './lib/util/matrix_math'

class Costs
  def mse_cost(data_y_hat, data_y)
    data_y = data_y.to_a
    data_y_hat = data_y_hat.to_a
    array = []
    i = 0
    while i < data_y_hat.size
      array[i] = 0.0
      j = 0
      while j < data_y_hat[i].size
        array[i] += (data_y[i][j] - data_y_hat[i][j])**2
        j += 1
      end
      array[i] = 0.5 * array[i] / data_y_hat[0].size
      i += 1
    end
    array.inject(:+) / data_y_hat.size
  end

  def crossentropy_cost(data_y_hat, data_y)
    data_y = data_y.to_a
    data_y_hat = data_y_hat.to_a
    array = []
    i = 0
    while i < data_y.size
      array[i] = 0.0
      j = 0
      while j < data_y[i].size
        array[i] -= data_y[i][j] * Math.log(data_y_hat[i][j] + 10**-8)
        j += 1
      end
      i += 1
    end
    array.inject(:+) / array.size
  end
end
