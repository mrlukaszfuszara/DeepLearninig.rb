class Costs
  def quadratic_cost(data_x, data_y)
    sum = 0
    i = 0
    while i < data_y.size
      sum += (data_y[i] - data_x[i])**2
      i += 1
    end
    sum = 0.5 * sum / data_x.size
    sum
  end

  def cross_entropy_cost(data_x, data_y)
    sum = 0
    i = 0
    while i < data_y.size
      sum += (data_y[i] * Math.log(data_x[i])) + ((1.0 - data_y[i]) * (Math.log(1.0 - data_x[i])))
      i += 1  
    end
    sum = -1.0 * sum / data_x.size
    sum
  end
end
