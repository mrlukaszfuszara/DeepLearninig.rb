class Matrix
  def []=(i, j, x)
    @rows[i][j] = x
  end
end

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

  def elementwise_div(matrix1, matrix2)
    if matrix1.row_size == matrix2.row_size && matrix1.column_size == matrix2.column_size
      array = []
      i = 0
      while i < matrix1.row_size
        array[i] = []
        j = 0
        while j < matrix1.column_size
          array[i][j] = matrix1[i, j] / matrix2[i, j]
          j += 1
        end
        i += 1
      end
    end
    Matrix[*array]
  end

  def elementwise_add(matrix1, variable)
    array = []
    i = 0
    while i < matrix1.row_size
      array[i] = []
      j = 0
      while j < matrix1.column_size
        array[i][j] = matrix1[i, j] + variable
        j += 1
      end
      i += 1
    end
    Matrix[*array]
  end
end
