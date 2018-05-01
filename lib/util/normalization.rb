class Normalization
  def initialize
    @mm = MatrixMath.new
  end

  def subt_mean(matrix)
    mean = []
    i = 0
    while i < matrix.size
      tmp = matrix[i].inject(:+)
      mean[i] = tmp / matrix.size
      i += 1
    end
    @mm.subt_reversed(matrix, mean)
  end

  def normalize_variance(matrix)
    mean = []
    i = 0
    while i < matrix.size
      tmp = matrix[i].inject(:+)
      mean[i] = tmp / matrix.size
      i += 1
    end
    sigma = []
    i = 0
    while i < matrix.size
      sigma[i] = []
      j = 0
      while j < matrix[i].size
        sigma[i][j] = (matrix[i][j] - mean[j])**2
        j += 1
      end
      i += 1
    end
    sigma_ready = []
    i = 0
    while i < sigma.size
      tmp = sigma[i].inject(:+)
      sigma_ready[i] = tmp / matrix.size
      i += 1
    end
    @mm.div_reversed(matrix, sigma_ready)
  end

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