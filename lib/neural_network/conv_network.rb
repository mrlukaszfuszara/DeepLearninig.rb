class ConvNetwork
  def initialize
    i = 0
    while i < matrix.size
      j = 0
      while j < matrix[i].size
        tmp = VectorizeArray.new
        matrix[i][j] = tmp.var_only(matrix[i][j])
        j += 1
      end
      i += 1
    end
    
    if filter.nil?
      filter = [[1.0] * start_size, [0.0] * start_size, [-1.0] * start_size]
      tmp = []
      i = 0
      while i < start_size
        tmp << filter
        i += 1
      end
      filter = tmp
    end
  end
end