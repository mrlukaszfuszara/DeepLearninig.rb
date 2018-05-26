class Matrix
  def []=(i, j, x)
    @rows[i][j] = x
  end

  def size
    [self.row_size, self.column_size]
  end

  def sqrt
    self.collect { |e| Math.sqrt e }
  end

  def pow(n)
    self.collect { |e| e**n }
  end

  def elementwise_matrix_div(matrix)
    i = 0
    while i < self.row_size
      j = 0
      while j < self.column_size
        self[i, j] /= matrix[i, j]
        j += 1
      end
      i += 1
    end
    self
  end

  def elementwise_var_div(var)
    i = 0
    while i < self.row_size
      j = 0
      while j < self.column_size
        self[i, j] /= var
        j += 1
      end
      i += 1
    end
    self
  end

  def elementwise_var_add(var)
    i = 0
    while i < self.row_size
      j = 0
      while j < self.column_size
        self[i, j] += var
        j += 1
      end
      i += 1
    end
    self
  end
end
