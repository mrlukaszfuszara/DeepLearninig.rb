class MatrixMath
  def matrix_dim(array)
    s0 = array.size
    s1 = array[0].size
    [s0, s1]
  end

  def f_norm(matrix)
    sum = 0
    array = []
    i = 0
    while i < matrix.size
      array[i] = 0
      j = 0
      while j < matrix[i].size
        array[i] += matrix[i][j]**2
        j += 1
      end
      sum += array[i]
      i += 1
    end
    sum
  end

  def add_reversed(variable1, variable2)
    one = matrix_check(variable1)
    two = matrix_check(variable2)
    array = []
    if one == 2 && two == 1
      variable1 = variable1.transpose
      if variable1[0].size == variable2.size
        i = 0
        while i < variable1.size
          array[i] = []
          j = 0
          while j < variable2.size
            array[i][j] = variable1[i][j] + variable2[j]
            j += 1
          end
          i += 1
        end
      end
    end
    array.transpose
  end

  def subt_reversed(variable1, variable2)
    one = matrix_check(variable1)
    two = matrix_check(variable2)
    array = []
    if one == 2 && two == 1
      variable1 = variable1.transpose
      if variable1[0].size == variable2.size
        i = 0
        while i < variable1.size
          array[i] = []
          j = 0
          while j < variable2.size
            array[i][j] = variable1[i][j] - variable2[j]
            j += 1
          end
          i += 1
        end
      end
    end
    array.transpose
  end

  def mult_reversed(variable1, variable2)
    one = matrix_check(variable1)
    two = matrix_check(variable2)
    array = []
    if one == 2 && two == 1
      variable1 = variable1.transpose
      if variable1[0].size == variable2.size
        i = 0
        while i < variable1.size
          array[i] = []
          j = 0
          while j < variable2.size
            array[i][j] = variable1[i][j] * variable2[j]
            j += 1
          end
          i += 1
        end
      end
    end
    array.transpose
  end

  def div_reversed(variable1, variable2)
    one = matrix_check(variable1)
    two = matrix_check(variable2)
    array = []
    if one == 2 && two == 1
      variable1 = variable1.transpose
      if variable1[0].size == variable2.size
        i = 0
        while i < variable1.size
          array[i] = []
          j = 0
          while j < variable2.size
            array[i][j] = variable1[i][j] / variable2[j]
            j += 1
          end
          i += 1
        end
      end
    end
    array.transpose
  end

  def matrix_abs(matrix)
    array = []
    i = 0
    while i < matrix.size
      array[i] = [0]
      j = 0
      while j < matrix[i].size
        array[i][j] = matrix[i][j].abs
        j += 1
      end
      i += 1
    end
    array
  end

  def add(variable1, variable2)
    one = matrix_check(variable1)
    two = matrix_check(variable2)
    array = []
    if one == 2 && two == 2
      if variable1.size == variable2.size && variable1[0].size == variable2[0].size
        i = 0
        while i < variable1.size
          array[i] = []
          j = 0
          while j < variable1[0].size
            array[i][j] = variable1[i][j] + variable2[i][j]
            j += 1
          end
          i += 1
        end
      else
        puts 'Add Matrix + Matrix: Size error'
      end
    elsif one == 2 && two == 1
      if variable1[0].size == variable2.size
        i = 0
        while i < variable1.size
          array[i] = []
          j = 0
          while j < variable2.size
            array[i][j] = variable1[i][j] + variable2[j]
            j += 1
          end
          i += 1
        end
      else
        puts 'Add Matrix + Vector: Size error'
      end
    elsif one == 2 && two.zero?
      i = 0
      while i < variable1.size
        array[i] = []
        j = 0
        while j < variable1[i].size
          array[i][j] = variable1[i][j] + variable2
          j += 1
        end
        i += 1
      end
    elsif one == 1 && two == 1
      if variable1.size == variable2.size
        i = 0
        while i < variable1.size
          array[i] = variable1[i] + variable2[i]
          i += 1
        end
      else
        puts 'Add Vector + Vector: Size error'
      end
    elsif one == 1 && two.zero?
      i = 0
      while i < variable1.size
        array[i] = variable1[i] + variable2
        i += 1
      end
    end
    array
  end

  def subt(variable1, variable2)
    one = matrix_check(variable1)
    two = matrix_check(variable2)
    array = []
    if one == 2 && two == 2
      if variable1.size == variable2.size && variable1[0].size == variable2[0].size
        i = 0
        while i < variable1.size
          array[i] = []
          j = 0
          while j < variable1[0].size
            array[i][j] = variable1[i][j] - variable2[i][j]
            j += 1
          end
          i += 1
        end
      else
        puts 'Subt Matrix - Matrix: Size error'
      end
    elsif one == 2 && two == 1
      if variable1[0].size == variable2.size
        i = 0
        while i < variable1.size
          array[i] = []
          j = 0
          while j < variable2.size
            array[i][j] = variable1[i][j] - variable2[j]
            j += 1
          end
          i += 1
        end
      else
        puts 'Subt Matrix - Vector: Size error'
      end
    elsif one == 2 && two.zero?
      i = 0
      while i < variable1.size
        array[i] = []
        j = 0
        while j < variable1[i].size
          array[i][j] = variable1[i][j] - variable2
          j += 1
        end
        i += 1
      end
    elsif one == 1 && two == 1
      if variable1.size == variable2.size
        i = 0
        while i < variable1.size
          array[i] = variable1[i] - variable2[i]
          i += 1
        end
      else
        puts 'Subt Vector - Vector: Size error'
      end
    elsif one == 1 && two.zero?
      i = 0
      while i < variable1.size
        array[i] = variable1[i] - variable2
        i += 1
      end
    end
    array
  end

  def mult(variable1, variable2)
    one = matrix_check(variable1)
    two = matrix_check(variable2)
    array = []
    if one == 2 && two == 2
      if variable1.size == variable2.size && variable1[0].size == variable2[0].size
        i = 0
        while i < variable1.size
          array[i] = []
          j = 0
          while j < variable1[0].size
            array[i][j] = variable1[i][j] * variable2[i][j]
            j += 1
          end
          i += 1
        end
      else
        puts 'Mult Matrix * Matrix: Size error'
      end
    elsif one == 2 && two == 1
      if variable1[0].size == variable2.size
        i = 0
        while i < variable1.size
          array[i] = []
          j = 0
          while j < variable2.size
            array[i][j] = variable1[i][j] * variable2[j]
            j += 1
          end
          i += 1
        end
      else
        puts 'Mult Matrix * Vector: Size error'
      end
    elsif one == 2 && two.zero?
      i = 0
      while i < variable1.size
        array[i] = []
        j = 0
        while j < variable1[i].size
          array[i][j] = variable1[i][j] * variable2
          j += 1
        end
        i += 1
      end
    elsif one == 1 && two == 1
      if variable1.size == variable2.size
        i = 0
        while i < variable1.size
          array[i] = variable1[i] * variable2[i]
          i += 1
        end
      else
        puts 'Mult Vector * Vector: Size error'
      end
    elsif one == 1 && two.zero?
      i = 0
      while i < variable1.size
        array[i] = variable1[i] * variable2
        i += 1
      end
    end
    array
  end

  def div(variable1, variable2)
    one = matrix_check(variable1)
    two = matrix_check(variable2)
    array = []
    if one == 2 && two == 2
      puts 'Div Error: Matrix / Matrix'
    elsif one == 2 && two == 1
      puts 'Div Error: Matrix / Vector'
    elsif one == 2 && two.zero?
      i = 0
      while i < variable1.size
        array[i] = []
        j = 0
        while j < variable1[i].size
          array[i][j] = variable1[i][j] / variable2
          j += 1
        end
        i += 1
      end
    elsif one == 1 && two == 1
      puts 'Div Error: Vector / Vector'
    elsif one == 1 && two.zero?
      i = 0
      while i < variable1.size
        array[i] = variable1[i] /+ variable2
        i += 1
      end
    end
    array
  end

  def dot(variable1, variable2)
    one = matrix_check(variable1)
    two = matrix_check(variable2)
    array = []
    if one == 2 && two == 2
      if variable1[0].size == variable2.size
        i = 0
        while i < variable1.size
          array[i] = []
          j = 0
          while j < variable2[0].size
            array[i][j] = 0
            k = 0
            while k < variable1[0].size
              array[i][j] += variable1[i][k] * variable2[k][j]
              k += 1
            end
            j += 1
          end
          i += 1
        end
      else
        puts 'Dot Matrix @ Matrix: Size error'
      end
    elsif one == 2 && two == 1
      if  variable1[0].size == variable2.size
        i = 0
        while i <  variable1.size
          array[i] = 0
          j = 0
          while j <  variable1[i].size
            array[i] +=  variable1[i][j] * variable2[j]
            j += 1
          end
          i += 1
        end
      else
        puts 'Dot Matrix @ Vector: Size error'
      end
    elsif one == 1 && two == 1
      if variable1.size == variable2.size
        array = 0
        i = 0
        while i < variable1.size
          array += variable1[i] * variable2[i]
          i += 1
        end
      else
        puts 'Dot Vector @ Vector: Size error'
      end
    end
    array
  end

  def vertical_sum(matrix)
    array = []
    i = 0
    while i < matrix.size
      array[i] = 0
      j = 0
      while j < matrix[i].size
        array[i] += matrix[i][j]
        j += 1
      end
      i += 1
    end
    array
  end

  def horizontal_sum(matrix)
    array = []
    i = 0
    while i < matrix[0].size
      array[i] = 0
      j = 0
      while j < matrix.size
        array[i] += matrix[j][i]
        j += 1
      end
      i += 1
    end
    array
  end

  private

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
