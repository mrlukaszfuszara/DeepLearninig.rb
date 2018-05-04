class Normalization
  attr_reader :mean, :sigma, :sd

  def initialize(calc = true, matrix = nil)
    @mm = MatrixMath.new
    if calc
      calculate(matrix)
    end
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
end
