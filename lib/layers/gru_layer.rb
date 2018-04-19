class GRULayer
  attr_reader :output

  def initialize(batch_size)
    @f = Functions.new
    @batch_size = batch_size
  end

  def compile_data
    create_weights
  end

  def fit_forward(input = nil)
    conc1 = calc_forward(input)
    layer_1_2 = apply_activation(conc1, 'sigmoid')
    step1 = @f.mult(@weights, layer_1_2)
    conc2 = @f.add(step1, input)
    layer_3 = apply_activation(conc2, 'tanh')
    step2 = @f.subt(layer_1_2, 1.0)
    step3 = @f.mult(@weights, step2)
    step4 = @f.mult(layer_1_2, layer_3)
    @weights = @f.add(step3, step4)
    @output = @weights
  end

  def create_weights
    @weights = @f.random_vector_full(@batch_size)
    @memory = @f.random_vector_full(@batch_size)
  end

  def apply_activation(layer, activation)
    tmp = 0
    if activation == 'sigmoid'
      tmp = @f.sigmoid(layer)
    elsif activation == 'tanh'
      tmp = @f.tanh(layer)
    elsif activation == 'relu'
      tmp = @f.relu(layer)
    end
    tmp
  end

  def calc_forward(input)
    @f.add(@weights, input)
  end
end
