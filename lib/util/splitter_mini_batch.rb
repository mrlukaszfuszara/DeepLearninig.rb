class SplitterMiniBatch
  attr_reader :x, :y

  def initialize(datx, daty, bs)
    x = []
    y = []
    i = bs
    while i < datx.size
      j = i - bs
      x[i] = []
      y[i] = []
      while j < i
        x[i][j] = datx[j]
        y[i][j] = daty[j]
        j += 1
      end
      x[i].compact!
      y[i].compact!
      i += bs
    end

    x.compact!
    y.compact!

    while x.size % bs != 0
      x.pop
      y.pop
    end

    @x = x
    @y = y
  end
end