class Costs
  def initialize
    @mm = MatrixMath.new
  end

  def quadratic_cost(data_y_hat, data_y)
    sum = 0
    i = 0
    while i < data_y.size
      tmp = @mm.subt(data_y[i], data_y_hat[i].flatten)
      sum += @mm.mult(tmp, tmp).inject(:+)
      i += 1
    end
    sum / data_y_hat.size
  end

  def quadratic_cost_with_r(data_y_hat, data_y, lambd, norm)
    sum = 0
    i = 0
    while i < data_y.size
      tmp = @mm.subt(data_y[i], data_y_hat[i].flatten)
      sum += @mm.add(@mm.mult(tmp, tmp), (lambd / (2.0 * data_y.size)) * norm).inject(:+)
      i += 1
    end
    sum / data_y_hat.size
  end
end
