class Costs
  def mse_cost(data_y_hat, data_y, lambd = nil, norm = nil)
    array = []
    i = 0
    while i < data_y_hat.size
      array[i] = []
      tmp = 0
      j = 0
      while j < data_y_hat[i].size
        tmp += (data_y[i][j] - data_y_hat[i][j])**2
        j += 1
      end
      array[i] = tmp / data_y_hat[0].size
      i += 1
    end
    if !lambd.nil?
      array = 0.5 * array.inject(:+) / data_y_hat.size + (lambd / (2.0 * data_y[i].size) * norm)
    else
      array = 0.5 * array.inject(:+) / data_y_hat.size
    end
    array
  end

  def crossentropy_cost(data_y_hat, data_y, lambd = nil, norm = nil)
    array = []
    i = 0
    while i < data_y_hat.size
      array[i] = []
      tmp = 0
      j = 0
      while j < data_y_hat[i].size
        tmp += -1.0 * data_y[i][j] * Math.log(data_y_hat[i][j])
        j += 1
      end
      array[i] = tmp
      i += 1
    end
    if !lambd.nil?
      array = array.inject(:+) / data_y_hat.size + (lambd / (2.0 * data_y[i].size) * norm)
    else
      array = array.inject(:+) / data_y_hat.size
    end
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
