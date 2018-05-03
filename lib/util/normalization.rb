class Normalization
  attr_reader :mean, :sigma

  def initialize(calc = true, matrix = nil)
    @mm = MatrixMath.new
    if calc
      calculate(matrix)
    end
  end

  def calculate(matrix)
    mean = @mm.horizontal_sum(matrix)
    i = 0
    while i < mean.size
      mean[i] = mean[i] / matrix.size
      i += 1
    end
    @mean = mean

    tmp1 = @mm.mult(matrix, matrix)
    sigma = @mm.horizontal_sum(tmp1)
    i = 0
    while i < sigma.size
      sigma[i] = sigma[i] / matrix.size
      i += 1
    end
    @sigma = sigma
  end

  def subt_mean(matrix, mean = nil)
    if !mean.nil?
      @mm.subt(matrix, mean)
    else
      @mm.subt(matrix, @mean)
    end
  end

  def normalize_x(matrix, sigma = nil)
    if !sigma.nil?
      @mm.div(matrix, @mm.vector_sqrt(@mm.add(sigma, 10**-8)))
    else
      @mm.div(matrix, @mm.vector_sqrt(@mm.add(@sigma, 10**-8)))
    end
  end

  def z_norm(matrix, mean = nil, sigma = nil)
    tmp1 = @mm.subt(matrix, mean)
    tmp2 = @mm.vector_sqrt(@mm.add(@sigma, 10**-8))
    @mm.div(tmp1, tmp2)
  end
end
