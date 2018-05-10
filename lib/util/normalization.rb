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

    @sd = @mm.vector_sqrt(@sigma)
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
    logic = matrix_check(matrix)
    if logic == 2
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
      tmp = array.transpose
    elsif logic == 1
      min_val = matrix.min
      max_val = matrix.max
      array = []
      i = 0
      while i < matrix.size
        array[i] = (matrix[i] - min_val) / (max_val - min_val)
        i += 1
      end
      tmp = array
    end
    tmp
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
