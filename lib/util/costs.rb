class Costs
  def initialize
    @mm = MatrixMath.new
  end

  def quadratic_cost(data_y_hat, data_y)
    array = []
    i = 0
    while i < data_y.size
      tmp = data_y[i] - data_y_hat[i].flatten[0]
      tmp = tmp**2
      array[i] = 0.5 * tmp / data_y.size
      i += 1
    end
    array.inject(:+) / array.size
  end

  def quadratic_cost_with_r(data_y_hat, data_y, lambd, norm)
    array = []
    i = 0
    while i < data_y.size
      tmp = data_y[i] - data_y_hat[i].flatten[0]
      tmp = tmp**2
      array[i] = 0.5 * tmp / data_y.size + (lambd / (2.0 * data_y.size) * norm)
      i += 1
    end
    array.inject(:+) / array.size
  end

  def log_loss_cost(data_y_hat, data_y)
    mln = @mm.matrix_ln(data_y_hat)
    array = []
    i = 0
    while i < data_y.size
      tmp = []
      j = 0
      while j < data_y[0].size
        tmp[i] = @mm.mult(data_y[i], mln[i])
        j += 1
      end
      tmp = @mm.mult(tmp[i], -1.0)
      array[i] = tmp.inject(:+)
      i += 1
    end
    tmp.inject(:+) / array.size
  end

  def log_loss_cost_with_r(data_y_hat, data_y, lambd, norm)
    mln = @mm.matrix_ln(data_y_hat)
    array = []
    i = 0
    while i < data_y.size
      tmp = []
      j = 0
      while j < data_y[0].size
        tmp[i] = @mm.mult(data_y[i], mln[i])
        j += 1
      end
      tmp = @mm.mult(tmp[i], -1.0)
      array[i] = tmp.inject(:+) + (lambd / (2.0 * data_y.size) * norm)
      i += 1
    end
    tmp.inject(:+) / array.size
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
