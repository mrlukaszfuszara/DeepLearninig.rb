class Costs
  def initialize
    @mm = MatrixMath.new
  end

  def quadratic_cost(data_y_hat, data_y)
    sum = 0
    i = 0
    while i < data_y.size
      tmp = data_y[i] - data_y_hat[i][0]
      sum += tmp**2
      i += 1
    end
    sum / data_y_hat[0].size
  end

  def quadratic_cost_with_r(data_y_hat, data_y, lambd, norm)
    sum = 0
    j = 0
    while j < data_y[0].size
      tmp = data_y[0][j] - data_y_hat[0][j][0]
      sum += tmp**2 + (lambd / (2.0 * data_y.size) * norm)
      j += 1
    end
    sum / data_y_hat[0].size
  end

  def log_loss_cost(data_y_hat, data_y)
    sum = 0
    i = 0
    while i < data_y[0].size
      tmp = 0
      j = 0
      while j < data_y[0][0].size
        tmp += data_y[0][i][j] * Math.log(data_y_hat[0][i][j])
        j += 1
      end
      sum += -1.0 * tmp
      i += 1
    end
    sum / data_y_hat[0].size
  end

  def log_loss_cost_with_r(data_y_hat, data_y, lambd, norm)
    sum = 0
    i = 0
    while i < data_y[0].size
      tmp = 0
      j = 0
      while j < data_y[0][0].size
        tmp += data_y[0][i][j] * Math.log(data_y_hat[0][i][j])
        j += 1
      end
      sum += -1.0 * tmp + (lambd / (2.0 * data_y.size) * norm)
      i += 1
    end
    sum / data_y_hat[0].size
  end
end
