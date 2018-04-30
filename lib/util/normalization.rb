class Normalization
  def initialize
    @mm = MatrixMath.new
  end

  def subt_mean(matrix)
    mean = @mm.mult(@mm.vertical_sum(matrix), (1.0 / matrix.size))
    @mm.subt_reversed(matrix, mean)
  end

  def normalize_variance(matrix)
    sigma_sqr = @mm.mult(@mm.vertical_sum(@mm.mult(matrix, 2.0)), (1.0 / matrix.size))
    @mm.div_reversed(matrix, sigma_sqr)
  end
end