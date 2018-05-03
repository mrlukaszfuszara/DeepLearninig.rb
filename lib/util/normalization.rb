class Normalization
  attr_reader :mean, :sigma

  def initialize
    @mm = MatrixMath.new
  end

  def subt_mean(matrix, mean = nil)
    if mean.nil?
      mean = @mm.horizontal_sum(matrix)
      i = 0
      while i < mean.size
        mean[i] = mean[i] / matrix.size
        i += 1
      end
      @mean = mean
      @mm.subt(matrix, mean)
    else
      @mm.subt(matrix, mean)
    end
  end

  def normalize_variance(matrix)
    
  end
end
