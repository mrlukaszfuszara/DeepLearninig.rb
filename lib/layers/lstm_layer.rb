class LSTMLayer
  attr_reader :output

  def initialize(batch_size)
    @f = Functions.new
    @batch_size = batch_size
  end

  def compile_data
    create_weights
  end

  def fit_forward(input = nil)
    conc = calc_forward(input)
    layer_1_2_4 = apply_activation(conc, 'sigmoid')
    layer_3 = apply_activation(conc, 'tanh')
    step1 = @f.mult(@memory, layer_1_2_4)
    step2 = @f.mult(layer_1_2_4, layer_3)
    step3 = @f.add(step1, step2)
    @memory = step3
    step4 = apply_activation(step3, 'tanh')
    step5 = @f.mult(layer_1_2_4, step4)
    @output = step5
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
