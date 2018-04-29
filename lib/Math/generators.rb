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
end
