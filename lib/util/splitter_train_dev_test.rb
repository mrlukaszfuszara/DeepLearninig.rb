class SpliterTrainDevTest
  attr_reader :train_s, :dev_s, :test_s

  def initialize(data_x, data_y, train_size = 0.6, dev_size = 0.2, test_size = 0.2)
    if train_size + dev_size + test_size == 1.0
      @data_x = data_x
      @data_y = data_y

      split_sets(train_size, dev_size, test_size)
    else
      "Error: Sets size don't match"
    end
  end

  def split_sets(train_size, dev_size, test_size)
    data_x_size = @data_x.size
    size_of_train = (data_x_size * train_size).floor
    size_of_dev = (data_x_size * dev_size).floor
    size_of_test = (data_x_size * test_size).floor

    @train_s = [[], []]
    @dev_s = [[], []]
    @test_s = [[], []]

    i = 0
    while i < data_x_size
      if i < size_of_train
        @train_s[0] << @data_x[i]
        @train_s[1] << @data_y[i]
      elsif i >= size_of_train && i < size_of_train + size_of_dev
        @dev_s[0] << @data_x[i]
        @dev_s[1] << @data_y[i]
      elsif i >= size_of_train + size_of_dev && i <= size_of_train + size_of_dev + size_of_test
        @test_s[0] << @data_x[i]
        @test_s[1] << @data_y[i]
      end
      i += 1
    end
  end

  def one_class_y(data_y)
    if !data_y.all? { |e| e.class == Array }
      data_y = [data_y].transpose
    end
    data_y
  end
end
