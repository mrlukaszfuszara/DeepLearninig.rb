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

  def random_volume(no_filters, channels, range)
    r = Random.new
    array = []
    filters = 0
    while filters < no_filters
      array[filters] = []
      tmp = r.rand(range)
      k = 0
      while k < channels
        array[filters] = tmp
        k += 1
      end
      filters += 1
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

  def tags_to_numbers(data_y)
    uniqe_array = []
    i = 0
    while i < data_y.size
      if !uniqe_array.include?(data_y[i])
        uniqe_array << data_y[i]
      end
      i += 1
    end
    labeled_array = []
    i = 0
    while i < data_y.size
      j = 0
      while j < uniqe_array.size
        if data_y[i] == uniqe_array[j]
          labeled_array << uniqe_array.index(uniqe_array[j])
        end
        j += 1
      end
      i += 1
    end
    labeled_array
  end
end
