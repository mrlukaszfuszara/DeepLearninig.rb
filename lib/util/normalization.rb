class Normalization
  attr_reader :mean, :sigma, :sd

  def initialize
    @mm = MatrixMath.new
  end

  def calculate(matrix)
    mean = @mm.horizontal_sum(matrix)
    @mean = @mm.div(mean, matrix.size)

    tmp = @mm.mult(matrix, matrix)
    tmp = @mm.horizontal_sum(tmp)
    sigma = @mm.subt(tmp, @mean)

    @sigma = @mm.div(tmp, matrix.size)

    @sd = @mm.vector_sqrt(@mm.add(@sigma, 10**-8))
  end

  def z_score(matrix, mean = nil, sd = nil)
    if !mean.nil? && !sd.nil?
      tmp = @mm.div(@mm.subt(matrix, mean), sd)
    else
      tmp = @mm.div(@mm.subt(matrix, @mean), @sd)
    end
    tmp
  end

  def min_max_scaler(matrix)
    matrix = matrix.transpose
    min_val = []
    max_val = []
    i = 0
    while i < matrix.size
      min_val[i] = matrix[i].min
      max_val[i] = matrix[i].max
      i += 1
    end
    array = []
    i = 0
    while i < matrix.size
      array[i] = []
      j = 0
      while j < matrix[i].size
        array[i][j] = (matrix[i][j] - min_val[i]) / (max_val[i] - min_val[i])
        j += 1
      end
      i += 1
    end
    array.transpose
  end
end
