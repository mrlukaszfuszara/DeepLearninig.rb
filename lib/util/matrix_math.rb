class MatrixMath
  def dim(array)
    s0 = array.size
    s1 = array[0].size
    p [s0, s1]
  end

  def matrix_size(array)
    arr = []
    i = 0
    while i < array.size
      arr << array[i].size
      i += 1
    end
    arr
  end

  def reverse_matrix(matrix)
    array = Array.new(matrix.size)
    i = 0
    while i < matrix.size
      array[i] = matrix[-i]
      i += 1
    end
    array
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

  def matrix_sqrt(matrix)
    array = []
    i = 0
    while i < matrix.size
      array[i] = []
      j = 0
      while j < matrix[i].size
        array[i][j] = Math.sqrt(matrix[i][j])
        j += 1
      end
      i += 1
    end
    array
  end

  def vector_sqrt(vector)
    array = []
    i = 0
    while i < vector.size
      if vector[i] >= 0
        array[i] = Math.sqrt(vector[i])
      else
        array[i] = 10**-8
      end
      i += 1
    end
    array
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

  def matrix_ln(matrix)
    array = []
    i = 0
    while i < matrix.size
      array[i] = [0]
      j = 0
      while j < matrix[i].size
        array[i][j] = Math.log(matrix[i][j])
        j += 1
      end
      i += 1
    end
    array
  end

  def vector_ln(vector)
    array = []
    i = 0
    while i < vector.size
      array[i] = Math.log(vector[i])
      i += 1
    end
    array
  end

  def matrix_exp(matrix)
    array = []
    i = 0
    while i < matrix.size
      array[i] = [0]
      j = 0
      while j < matrix[i].size
        array[i][j] = Math.exp(matrix[i][j])
        j += 1
      end
      i += 1
    end
    array
  end

  def vector_exp(vector)
    array = []
    i = 0
    while i < vector.size
      array[i] = Math.exp(vector[i])
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
          while j < variable1[0].size
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
          while j < variable1[0].size
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
          while j < variable1[0].size
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
      if variable1.size == variable2.size && variable1[0].size == variable2[0].size
        i = 0
        while i < variable1.size
          array[i] = []
          j = 0
          while j < variable1[0].size
            array[i][j] = variable1[i][j] / variable2[i][j]
            j += 1
          end
          i += 1
        end
      else
        puts 'Div Matrix / Matrix: Size error'
      end
    elsif one == 2 && two == 1
      if variable1[0].size == variable2.size
        i = 0
        while i < variable1.size
          array[i] = []
          j = 0
          while j < variable1[0].size
            array[i][j] = variable1[i][j] / variable2[j]
            j += 1
          end
          i += 1
        end
      else
        puts 'Div Matrix / Vector: Size error'
      end
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
      if variable1.size == variable2.size
        i = 0
        while i < variable1.size
          array[i] = variable1[i] / variable2[i]
          i += 1
        end
      else
        puts 'Div Vector / Vector: Size error'
      end
    elsif one == 1 && two.zero?
      i = 0
      while i < variable1.size
        array[i] = variable1[i] / variable2
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
          while j <  variable1[0].size
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
    matrix = matrix.transpose

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

  def conv2d(matrix, orient, padding = true, filter = nil)
    array = []
    if filter.nil?
      if orient == 'h'
        filter = [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]]
      elsif orient == 'v'
        filter = [[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]
      end
    end
    if padding
      i = 0
      while i < matrix.size
        matrix[i].push(0.0)
        matrix[i].unshift(0.0)
        i += 1
      end
      matrix.push(Array.new(matrix.size, 0.0))
      matrix.unshift(Array.new(matrix.size, 0.0))
    end
    if filter.size.to_f % 2 != 0
      array = []
      i = 0
      while i <= (matrix.size / 2.0).floor
        array[i] = []
        tmp = []
        j = 0
        while j <= (matrix[i].size / 2.0).floor
          tmp[i] = 0
          k = 0
          while k < filter.size
            l = 0
            while l < filter[0].size
              tmp[i] += (matrix[i + k][j + l] * filter[k][l]).floor
              l += 1
            end
            k += 1
          end
          array[i][j] = tmp[i]
          j += 1
        end
        i += 1
      end
    else
      puts 'Conv2D: Filter size error'
    end
    array
  end

  def conv3d(matrix, orient, padding = true, filter = nil)
    array = []
    if filter.nil?
      if orient == 'h'
        filter = [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]]
      elsif orient == 'v'
        filter = [[[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]], [[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]], [[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]]
      end
    end
    if padding
      i = 0
      while i < matrix.size
        matrix[i].push([0.0] * matrix[0][0].size)
        matrix[i].unshift([0.0] * matrix[0][0].size)
        i += 1
      end
      matrix.push(Array.new(matrix.size, [0.0] * matrix[0][0].size))
      matrix.unshift(Array.new(matrix.size, [0.0] * matrix[0][0].size))
    end
    if filter.size.to_f % 2 != 0
      array = []
      i = 0
      while i <= (matrix.size / 2.0).floor
        array[i] = []
        tmp = []
        j = 0
        while j <= (matrix[i].size / 2.0).floor
          tmp[i] = 0
          ch = 0
          while ch < matrix[i][j].size
            k = 0
            while k < filter.size
              l = 0
              while l < filter[0].size
                tmp[i] += (matrix[i + k][j + l][ch] * filter[k][l][ch]).floor
                l += 1
              end
              k += 1
            end
            ch += 1
          end
          array[i][j] = tmp[i]
          j += 1
        end
        i += 1
      end
    else
      puts 'Conv3D: Filter size error'
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
