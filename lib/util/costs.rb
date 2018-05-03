class Costs
  def quadratic_cost(data_y_hat, data_y, data_x_size)
    sum = 0
    i = 0
    while i < data_y.size
      sum += (data_y[i] - data_y_hat[i])**2
      i += 1
    end
    sum / data_x_size
  end

  def quadratic_cost_with_r(data_y_hat, data_y, data_x_size, lambd, norm)
    sum = 0
    i = 0
    while i < data_y.size
      sum += (data_y[i] - data_y_hat[i])**2 + (lambd / (2.0 * data_x_size)) * norm
      i += 1
    end
    sum / data_x_size
  end

  def cross_entropy_cost(data_x, data_y, data_x_size)
    sum = 0
    i = 0
    while i < data_y.size
      sum += (data_y[i] * Math.log(data_x[i])) + ((1.0 - data_y[i]) * (Math.log(1.0 - data_x[i])))
      i += 1  
    end
    (-1.0 * sum) / data_x_size
  end

  def cross_entropy_cost_with_r(data_x, data_y, data_x_size, lambd, norm)
    sum = 0
    i = 0
    while i < data_y.size
      sum += (data_y[i] * Math.log(data_x[i])) + ((1.0 - data_y[i]) * (Math.log(1.0 - data_x[i]))) + (lambd / (2.0 * data_x_size)) * norm
      i += 1  
    end
    (-1.0 * sum) / data_x_size
  end
end
