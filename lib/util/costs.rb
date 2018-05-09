class Costs
  def initialize
    @mm = MatrixMath.new
  end

  def mse_cost(data_y_hat, data_y, lambd = nil, norm = nil)
    tmp = Array.new(data_y_hat[0].size, 0.0)
    i = 0
    while i < data_y_hat.size
      j = 0
      while j < data_y_hat[i].size
        tmp[j] += (data_y[i][j] - data_y_hat[i][j])**2
        j += 1
      end
      i += 1
    end
    i = 0
    while i < tmp.size
      if !lambd.nil?
        tmp[i] = 0.5 * tmp[i] / data_y_hat[i].size + (lambd / (2.0 * data_y.size) * norm)
      else
        tmp[i] = 0.5 * tmp[i] / data_y_hat[i].size
      end
      i += 1
    end
    tmp.inject(:+) / data_y_hat.size
  end

  def crossentropy_cost(data_y_hat, data_y, lambd = nil, norm = nil)
    tmp = Array.new(data_y_hat[0].size, 0.0)
    i = 0
    while i < data_y_hat.size
      j = 0
      while j < data_y_hat[i].size
        tmp[j] += (data_y[i][j] * Math.log(data_y_hat[i][j])) + (1.0 - data_y[i][j]) * Math.log(1.0 - data_y_hat[i][j])
        j += 1
      end
      i += 1
    end
    i = 0
    while i < tmp.size
      if !lambd.nil?
        tmp[i] = -1.0 * tmp[i] / data_y_hat[i].size + (lambd / (2.0 * data_y.size) * norm)
      else
        tmp[i] = -1.0 * tmp[i] / data_y_hat[i].size
      end
      i += 1
    end
    tmp.inject(:+) / data_y_hat.size
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
