class Costs
  def initialize
    @mm = MatrixMath.new
  end

  def quadratic_cost(data_y_hat, data_y)
    array = []
    i = 0
    while i < data_y[0].size
      tmp = @mm.subt(data_y[i], data_y_hat.flatten)
      tmp = @mm.mult(tmp, tmp)
      tmp = 0.5 * tmp.inject(:+)
      i += 1
    end
    tmp
  end

  def quadratic_cost_with_r(data_y_hat, data_y, lambd, norm)
    array = []
    i = 0
    while i < data_y[0].size
      tmp = @mm.subt(data_y[i], data_y_hat.flatten)
      tmp = @mm.mult(tmp, tmp)
      tmp = 0.5 * tmp.inject(:+) + (lambd / (2.0 * data_y.size) * norm)
      i += 1
    end
    tmp
  end

  def log_loss_cost(data_y_hat, data_y)
    array = []
    i = 0
    while i < data_y[0].size
      tmp = @mm.mult(@mm.mult(data_y[i], @mm.vector_ln(data_y_hat[i])), -1.0)
      tmp = tmp.inject(:+)
      i += 1
    end
    tmp.inject(:+) / data_y[0].size
  end

  def log_loss_cost_with_r(data_y_hat, data_y, lambd, norm)
    array = []
    i = 0
    while i < data_y[0].size
      tmp = @mm.mult(@mm.mult(data_y[i], @mm.vector_ln(data_y_hat[i])), -1.0)
      tmp = tmp.inject(:+) + (lambd / (2.0 * data_y.size) * norm)
      i += 1
    end
    tmp.inject(:+) / data_y[0].size
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
