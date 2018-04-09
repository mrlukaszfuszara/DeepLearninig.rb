class Flatten
  def initialize
  end

  def compile_data(data)
  end

  def fit_forward(data_x)
  	data_x
  end

  def fit_backward(weights, delta, output, data_y = nil, last_layer = nil)
    data_x.flatten
  end
end