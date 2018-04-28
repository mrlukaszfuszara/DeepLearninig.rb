class Activations
  def sigmoid(vector)
    array = []
    v = matrix_check(vector)
    if v == 2
      i = 0
      while i < vector.size
        array[i] = []
        j = 0
        while j < vector[i].size
          array[i][j] = 1.0 / (1.0 + Math.exp(-vector[i][j]))
          j += 1
        end
        i += 1
      end
    elsif v == 1
      i = 0
      while i < vector.size
        array[i] = 1.0 / (1.0 + Math.exp(-vector[i]))
        i += 1
      end
    end
    array
  end

  def sigmoid_d(vector)
    array = []
    v = matrix_check(vector)
    if v == 2
      i = 0
      while i < vector.size
        array[i] = []
        j = 0
        while j < vector[i].size
          array[i][j] = vector[i][j] * (1.0 - vector[i][j])
          j += 1
        end
        i += 1
      end
    elsif v == 1
      i = 0
      while i < vector.size
        array[i] = vector[i] * (1.0 - vector[i])
        i += 1
      end
    end
    array
  end

  def tanh(vector)
    array = []
    v = matrix_check(vector)
    if v == 2
      i = 0
      while i < vector.size
        array[i] = []
        j = 0
        while j < vector[i].size
          array[i][j] = (Math.exp(vector[i][j]) - Math.exp(-vector[i][j])) / (Math.exp(vector[i][j]) + Math.exp(-vector[i][j]))
          j += 1
        end
        i += 1
      end
    elsif v == 1
      i = 0
      while i < vector.size
        array[i] = (Math.exp(vector[i]) - Math.exp(-vector[i])) / (Math.exp(vector[i]) + Math.exp(-vector[i]))
        i += 1
      end
    end
    array
  end

  def tanh_d(vector)
    array = []
    v = matrix_check(vector)
    if v == 2
      i = 0
      while i < vector.size
        array[i] = []
        j = 0
        while j < vector[i].size
          array[i][j] = 1.0 - Math.tanh(vector[i][j])**2
          j += 1
        end
        i += 1
      end
    elsif v == 1
      i = 0
      while i < vector.size
        array[i] = 1.0 - Math.tanh(vector[i])**2
        i += 1
      end
    end
    array
  end

  def relu(vector)
    array = []
    v = matrix_check(vector)
    if v == 2
      i = 0
      while i < vector.size
        array[i] = []
        j = 0
        while j < vector[i].size
          if vector[i][j] > 0.0
            array[i][j] = vector[i][j]
          else
            array[i][j] = 0.0
          end
          j += 1
        end
        i += 1
      end
    elsif v == 1
      i = 0
      while i < vector.size
        if vector[i] > 0.0
          array[i] = vector[i]
        else
          array[i] = 0.0
        end
        i += 1
      end
    end
    array
  end

  def relu_d(vector)
    array = []
    v = matrix_check(vector)
    if v == 2
      i = 0
      while i < vector.size
        array[i] = []
        j = 0
        while j < vector[i].size
          if vector[i][j] > 0.0
            array[i][j] = 1.0
          else
            array[i][j] = 0.0
          end
          j += 1
        end
        i += 1
      end
    elsif v == 1
      i = 0
      while i < vector.size
        if vector[i] > 0.0
          array[i] = 1.0
        else
          array[i] = 0.0
        end
        i += 1
      end
    end
    array
  end

  def leaky_relu(vector)
    array = []
    v = matrix_check(vector)
    if v == 2
      i = 0
      while i < vector.size
        array[i] = []
        j = 0
        while j < vector[i].size
          tmp = 0.01 * vector[i][j]
          if vector[i][j] > tmp
            array[i][j] = vector[i][j]
          else
            array[i][j] = tmp
          end
          j += 1
        end
        i += 1
      end
    elsif v == 1
      i = 0
      while i < vector.size
        tmp = 0.01 * vector[i]
        if vector[i] > tmp
          array[i] = vector[i]
        else
          array[i] = tmp
        end
        i += 1
      end
    end
    array
  end

  def leaky_relu_d(vector)
    array = []
    v = matrix_check(vector)
    if v == 2
      i = 0
      while i < vector.size
        array[i] = []
        j = 0
        while j < vector[i].size
          if vector[i][j] > 0.0
            array[i][j] = 1.0
          else
            array[i][j] = 0.01
          end
          j += 1
        end
        i += 1
      end
    elsif v == 1
      i = 0
      while i < vector.size
        if vector[i] > 0.0
          array[i] = 1.0
        else
          array[i] = 0.01
        end
        i += 1
      end
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
