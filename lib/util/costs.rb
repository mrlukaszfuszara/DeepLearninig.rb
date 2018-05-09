class Costs
  def initialize
    @mm = MatrixMath.new
  end

  def mse_cost(data_y_hat, data_y, lambd = nil, norm = nil)
    tmp = 0
    i = 0
    while i < data_y_hat.size
      tmp += (data_y[i] - data_y_hat[i][0])**2
      i += 1
    end
    if !lambd.nil?
      tmp = 0.5 * tmp / data_y_hat.size + (lambd / (2.0 * data_y.size) * norm)
    else
      tmp = 0.5 * tmp / data_y_hat.size
    end
    tmp
  end

  def crossentropy_cost(data_y_hat, data_y, lambd = nil, norm = nil)
    tmp = 0
    i = 0
    while i < data_y_hat.size
      tmp += (data_y[i] * Math.log(data_y_hat[i][0])) + (1.0 - data_y[i]) * Math.log(1.0 - data_y_hat[i][0])
      i += 1
    end
    if !lambd.nil?
      tmp = -1.0 * tmp / data_y_hat.size + (lambd / (2.0 * data_y.size) * norm)
    else
      tmp = -1.0 * tmp / data_y_hat.size
    end
    tmp
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
