class Costs
  def initialize
    @mm = MatrixMath.new
  end

  def quadratic_cost(data_y_hat, data_y)
    tmp = 0
    i = 0
    while i < data_y_hat.size
      tmp += (data_y[i] - data_y_hat[i][0])**2
      i += 1
    end
    0.5 * tmp / data_y_hat.size
  end

  def quadratic_cost_with_r(data_y_hat, data_y, lambd, norm)
    tmp = 0
    i = 0
    while i < data_y_hat.size
      tmp += (data_y[i] - data_y_hat[i][0])**2
      i += 1
    end
    0.5 * tmp / data_y_hat.size + (lambd / (2.0 * data_y.size) * norm)
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
