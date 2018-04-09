class Dropout
  def initialize(rate)
    @rate = rate / 2.0
  end

  def compile_data(data)
    @r = Random.new
  end

  def fit_forward(data_x)
    array = Array.new
    i = 0
    while i < data_x.size
      array[i] = data_x[i]  + (@r.rand(-@rate..@rate)).to_f
      i += 1
    end
    array
  end

  def fit_backward(weights, delta, output, data_y = nil, last_layer = nil)
  	output
  end
end