class MatrixMath
  def dim(matrix)
    [matrix.column_size, matrix.row_size]
  end

  def matrix_sqrt(matrix)
    matrix.collect { |e| Math.sqrt e }
  end

  def matrix_ln(matrix)
    matrix.collect { |e| Math.log e }
  end

  def matrix_exp(matrix)
    matrix.collect { |e| Math.exp e }
  end

  def matrix_square(matrix)
    matrix.collect { |e| e**2 }
  end
end
