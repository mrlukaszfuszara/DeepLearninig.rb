class Normalization
  def initialize
    @mm = MatrixMath.new
  end

  def subt_mean(matrix)
    l = matrix_check(matrix)
    if l == 2
      mean = @mm.mult(@mm.vertical_sum(matrix), (1.0 / matrix.size))
      tmp = @mm.subt_reversed(matrix, mean)
    elsif l == 1
      mean = matrix.inject(:+) * (1.0 / matrix.size)
      tmp = @mm.subt(matrix, mean)
    end
    tmp
  end

  def normalize_variance(matrix)
    sigma_sqr = @mm.mult(@mm.vertical_sum(@mm.mult(matrix, 2.0)), (1.0 / matrix.size))
    @mm.div_reversed(matrix, sigma_sqr)
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