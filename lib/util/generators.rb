class Generators
  def random_matrix(size_rows, size_cols, range)
    r = Random.new
    array = []
    i = 0
    while i < size_rows
      array[i] = []
      j = 0
      while j < size_cols
        array[i][j] = r.rand(range)
        j += 1
      end
      i += 1
    end
    array
  end

  def random_vector(size_rows, range)
    r = Random.new
    array = []
    i = 0
    while i < size_rows
      array[i] = r.rand(range)
      i += 1
    end
    array
  end

  def zero_matrix(size_rows, size_cols)
    r = Random.new
    array = []
    i = 0
    while i < size_rows
      array[i] = []
      j = 0
      while j < size_cols
        array[i][j] = 0.0
        j += 1
      end
      i += 1
    end
    array
  end

  def zero_vector(size_rows)
    r = Random.new
    array = []
    i = 0
    while i < size_rows
      array[i] = 0.0
      i += 1
    end
    array
  end

  def one_matrix(size_rows, size_cols)
    r = Random.new
    array = []
    i = 0
    while i < size_rows
      array[i] = []
      j = 0
      while j < size_cols
        array[i][j] = 1.0
        j += 1
      end
      i += 1
    end
    array
  end

  def one_vector(size_rows)
    r = Random.new
    array = []
    i = 0
    while i < size_rows
      array[i] = 1.0
      i += 1
    end
    array
  end

  def dotzeroone_matrix(size_rows, size_cols)
    r = Random.new
    array = []
    i = 0
    while i < size_rows
      array[i] = []
      j = 0
      while j < size_cols
        array[i][j] = 0.01
        j += 1
      end
      i += 1
    end
    array
  end

  def dotzeroone_vector(size_rows)
    r = Random.new
    array = []
    i = 0
    while i < size_rows
      array[i] = 0.01
      i += 1
    end
    array
  end

  def one_hot_vector(data_y)
    min_val = data_y.min
    max_val = data_y.max
    max_subt_min = max_val - min_val
    array = []
    i = 0
    while i < data_y.size
      array[i] = []
      j = 0
      while j < max_subt_min
        if data_y[i] == j + min_val
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
end
