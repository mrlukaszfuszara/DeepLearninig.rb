class Functions
  def sigmoid(vector)
    array = Array.new
    i = 0
    while i < vector.size
      array[i] = 1.0 / (1.0 + Math.exp(-vector[i]))
      i += 1
    end
    array
  end

  def sigmoid_d(matrix)
    array = Array.new
    i = 0
    while i < matrix.size
      array[i] = []
      j = 0
      while j < matrix[i].size
        array[i][j] = matrix[i][j] * (1.0 - matrix[i][j])
        j += 1
      end
      i += 1
    end
    array
  end

  def tanh(vector)
    array = Array.new
    i = 0
    while i < vector.size
      array[i] = Math.tanh(vector[i])
      i += 1
    end
    array
  end

  def tanh_d(matrix)
    array = Array.new
    i = 0
    while i < matrix.size
      array[i] = []
      j = 0
      while j < matrix[i].size
        array[i][j] = Math.sinh(matrix[i][j]) / Math.cosh(matrix[i][j])
        j += 1
      end
      i += 1
    end
    array
  end

  def relu(vector)
    array = Array.new
    i = 0
    while i < vector.size
      if vector[i] > 0.0
        array[i] = vector[i]
      else
        array[i] = 0.0
      end
      i += 1
    end
    array
  end

  def relu_d(matrix)
    array = Array.new
    i = 0
    while i < matrix.size
      array[i] = []
      j = 0
      while j < matrix[i].size
        if matrix[i][j] > 0
          array[i][j] = 1.0
        else
          array[i][j] = 0.0
        end
          j += 1
        end
      i += 1
    end
    array
  end

  def mse_error(data, data_y)
    array = Array.new
    sum = 0
    i = 0
    while i < data_y.size
      sum += (data[i] - data_y[i])**2
      i += 1      
    end
    sum = 0.5 * sum
    sum
  end

  def dot(variable_1, variable_2)
    one = matrix_check(variable_1)
    two = matrix_check(variable_2)
    array = Array.new

    if one == 2 && two == 2
      if variable_1[0].size == variable_2.size
        i = 0
        while i < variable_1.size
          j = 0
          while j < variable_2[0].size
            tmp = 0
            k = 0
            while k < variable_1[0].size
              tmp += variable_1[i][k] * variable_2[k][j]
              k += 1
            end
            j += 1
          end
          array[i] = tmp
          i += 1
        end
      else
        puts "Dot Matrix @ Matrix: Size error"
        exit 1
      end
    elsif one == 2 && two == 1
      if  variable_1[0].size == variable_2.size
        i = 0
        while i <  variable_1.size
          array[i] = 0
          j = 0
          while j <  variable_1[i].size
            array[i] +=  variable_1[i][j] * variable_2[j]
            j += 1
          end
          i += 1
        end
      else
        puts "Dot Matrix @ Vector: Size error"
        exit 1
      end
    elsif one == 1 && two == 1
      if variable_1.size == variable_2.size
        array = 0
        i = 0
        while i < variable_1.size
          array += variable_1[i] * variable_2[i]
          i += 1
        end
      else
        puts "Dot Vector @ Vector: Size error"
        exit 1
      end
    end
        
    array
    end

  def add(variable_1, variable_2)
    one = matrix_check(variable_1)
    two = matrix_check(variable_2)
    array = Array.new
    if one == 2 && two == 2
      if variable_1[0].size == variable_2.size
      i = 0
      while i < variable_1.size
        array[i] = []
        j = 0
        while j < variable_2.size
          array[i][j] = variable_1[i][j] + variable_2[i][j]
          j += 1
        end
        i += 1
      end
      else
        puts "Add Matrix + Matrix: Size error"
        exit 1
      end
    elsif one == 2 && two == 1
      if variable_1[0].size == variable_2.size
      i = 0
      while i < variable_1.size
        array[i] = []
        j = 0
        while j < variable_2.size
          array[i][j] = variable_1[i][j] + variable_2[j]
          j += 1
        end
        i += 1
      end
     else
        puts "Add Matrix + Vector: Size error"
        exit 1
      end
    elsif one == 2 && two == 0
      i = 0
      while i < variable_1.size
        array[i] = []
        j = 0
        while j < variable_1[i].size
          array[i][j] = variable_1[i][j] + variable_2
          j += 1
        end
        i += 1
      end
    elsif one == 1 && two == 1
      if variable_1.size == variable_2.size
      i = 0
      while i < variable_1.size
        array[i] = variable_1[i] + variable_2[i]
        i += 1
      end
      else
        puts "Add Vector + Vector: Size error"
        exit 1
      end
    elsif one == 1 && two == 0
      i = 0
      while i < variable_1.size
        array[i] = variable_1[i] + variable_2
        i += 1
      end
    end
    array
  end

  def sub(variable_1, variable_2)
    one = matrix_check(variable_1)
    two = matrix_check(variable_2)
    array = Array.new
    if one == 2 && two == 2
      if variable_1[0].size == variable_2.size
      i = 0
      while i < variable_1.size
        array[i] = []
        j = 0
        while j < variable_2.size
          array[i][j] = variable_1[i][j] - variable_2[i][j]
          j += 1
        end
        i += 1
      end
      else
        puts "Sub Matrix - Matrix: Size error"
        exit 1
      end
    elsif one == 2 && two == 1
      if variable_1[0].size == variable_2.size
      i = 0
      while i < variable_1.size
        array[i] = []
        j = 0
        while j < variable_2.size
          array[i][j] = variable_1[i][j] - variable_2[j]
          j += 1
        end
        i += 1
      end
     else
        puts "Sub Matrix - Vector: Size error"
        exit 1
      end
    elsif one == 2 && two == 0
      i = 0
      while i < variable_1.size
        array[i] = []
        j = 0
        while j < variable_1[i].size
          array[i][j] = variable_1[i][j] - variable_2
          j += 1
        end
        i += 1
      end
    elsif one == 1 && two == 1
      if variable_1.size == variable_2.size
      i = 0
      while i < variable_1.size
        array[i] = variable_1[i] - variable_2[i]
        i += 1
      end
      else
        puts "Sub Vector - Vector: Size error"
        exit 1
      end
    elsif one == 1 && two == 0
      i = 0
      while i < variable_1.size
        array[i] = variable_1[i] - variable_2
        i += 1
      end
    end
    array
  end

  def mult(variable_1, variable_2)
    one = matrix_check(variable_1)
    two = matrix_check(variable_2)
    array = Array.new
    if one == 2 && two == 2
      if variable_1[0].size == variable_2.size
      i = 0
      while i < variable_1.size
        array[i] = []
        j = 0
        while j < variable_2.size
          array[i][j] = variable_1[i][j] * variable_2[i][j]
          j += 1
        end
        i += 1
      end
      else
        puts "Mult Matrix * Matrix: Size error"
        exit 1
      end
    elsif one == 2 && two == 1
      if variable_1[0].size == variable_2.size
      i = 0
      while i < variable_1.size
        array[i] = []
        j = 0
        while j < variable_2.size
          array[i][j] = variable_1[i][j] * variable_2[j]
          j += 1
        end
        i += 1
      end
     else
        puts "Mult Matrix * Vector: Size error"
        exit 1
      end
    elsif one == 2 && two == 0
      i = 0
      while i < variable_1.size
        array[i] = []
        j = 0
        while j < variable_1[i].size
          array[i][j] = variable_1[i][j] * variable_2
          j += 1
        end
        i += 1
      end
    elsif one == 1 && two == 1
      if variable_1.size == variable_2.size
      i = 0
      while i < variable_1.size
        array[i] = variable_1[i] * variable_2[i]
        i += 1
      end
      else
        puts "Mult Vector * Vector: Size error"
        exit 1
      end
    elsif one == 1 && two == 0
      i = 0
      while i < variable_1.size
        array[i] = variable_1[i] * variable_2
        i += 1
      end
    end
    array
  end

  def random_matrix_small(size_rows, size_cols)
    r = Random.new
    array = Array.new
    i = 0
    while i < size_rows
      array[i] = []
      j = 0
      while j < size_cols
        array[i][j] = r.rand(0.0..0.1)
        j += 1
      end
      i += 1
    end
    array
  end

  def random_matrix_full(size_rows, size_cols)
    r = Random.new
    array = Array.new
    i = 0
    while i < size_rows
      array[i] = []
      j = 0
      while j < size_cols
        array[i][j] = r.rand(0.0..1.0)
        j += 1
      end
      i += 1
    end
    array
  end

  def slice_vector(vector)
    array = Array.new
    i = 0
    while i < vector.size
      array[i] = [vector[i]]
      i += 1
    end
    array
  end

  def matrix_dim(array)
    s_0 = array.size
    s_1 = array[0].size
    [s_0, s_1]
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