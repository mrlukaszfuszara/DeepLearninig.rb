class SplitterMB
  attr_reader :data_x, :data_y

  def initialize(mini_batch_size, data_x, data_y)
    @x = data_x
    @y = data_y

    split_sets(mini_batch_size)
  end

  def split_sets(mini_batch_size)
    array_x = []
    array_y = []
    i = 0
    while i < @x.size
      array_x << @x[i..(i + mini_batch_size)]
      array_y << @y[i..(i + mini_batch_size)]
      i += mini_batch_size
    end
    if array_x.last.size != @x.size
      array_x.pop
      array_y.pop
    end
    @data_x = array_x
    @data_y = array_y
  end
end