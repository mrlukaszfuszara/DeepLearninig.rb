require './lib/util/matrix_math'

class Normalization
  attr_reader :matrix

  def initialize(matrix)
    @matrix = matrix
  end

  def min_max_scaler
    i = 0
    while i < @matrix.size
      @matrix[i] = @matrix[i].transpose
      j = 0
      while j < @matrix[i].size
        min_val = @matrix[i][j].min
        max_val = @matrix[i][j].max
        k = 0
        while k < @matrix[i][j].size
          @matrix[i][j][k] = (@matrix[i][j][k] - min_val) / (max_val - min_val)
          k += 1
        end
        j += 1
      end
      @matrix[i] = @matrix[i].transpose
      i += 1
    end
  end

  def z_score
    mean = []
    std_dev = []
    i = 0
    while i < @matrix.size
      @matrix[i] = @matrix[i].transpose
      mean[i] = @matrix[i].map { |e| e.inject(:+) / e.size }
      std_dev[i] = @matrix[i].map.with_index { |e, j| Math.sqrt(e.inject { |s, f| s + (f - mean[i][j])**2 } / (e.size - 1.0)) }
      @matrix[i] = @matrix[i].map.with_index { |e, j| e.map { |f| (f - mean[i][j]) / (std_dev[i][j] / Math.sqrt(e.size)) } }
      @matrix[i] = @matrix[i].transpose
      i += 1
    end
  end
end
