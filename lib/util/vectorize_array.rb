class VectorizeArray
  attr_reader :output

  def all(input)
    logic = matrix_check(input)
    if logic == 1
      input = [input].transpose
    elsif logic.zero?
      input = [input]
    end
    @output = input
  end

  def var_only(input)
    logic = matrix_check(input)
    if logic.zero?
      input = [input]
    end
    @output = input
  end

  def matrix_check(variable)
    logic = nil
    if variable.class == Array
      if variable.all? { |e| e.class == Array }
        logic = 2
      else
        logic = 1
      end
    else
      logic = 0
    end
    logic
  end
end