class RNNLayer
  attr_reader :output

  def initialize(batch_size)
    @f = Functions.new
    @batch_size = batch_size
  end

  def compile_data
    create_weights
  end

  def fit_forward(input = nil)
    @output = calc_forward(input)
    @output = apply_activation(@output)
  end

  def create_weights
    @weights = @f.random_vector_full(@batch_size)
  end

  def apply_activation(layer)
    @f.tanh(layer)
  end

  def calc_forward(input)
    @f.add(@weights, input)
  end
end
