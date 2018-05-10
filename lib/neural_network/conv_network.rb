class ConvNetwork
  def initialize
    @cm = ConvMath.new
  end

  def input(image)
    i = 0
    while i < image.size
      j = 0
      while j < image[i].size
        tmp = VectorizeArray.new
        image[i][j] = tmp.var_only(image[i][j])
        j += 1
      end
      i += 1
    end

    img_size = image[0][0].size

    filter = nil
    if filter.nil?
      filter = [[1.0] * img_size, [0.0] * img_size, [-1.0] * img_size]
      tmp = []
      i = 0
      while i < img_size
        tmp << filter
        i += 1
      end
      filter = tmp
    end

    p @cm.conv2d(image, filter)
  end
end
