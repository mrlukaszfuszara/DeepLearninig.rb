require 'chunky_png'

class ImageLoader
  def load_image(image_path)
    img = ChunkyPNG::Image.from_file(image_path)

    height = img.dimension.height
    width  = img.dimension.width

    red_array = []
    green_array = []
    blue_array = []

    height.times do |i|
      red_array[i] = []
      green_array[i] = []
      blue_array[i] = []
      width.times do |j|
        red_array[i] << ChunkyPNG::Color.r(img[j, i])
        green_array[i] << ChunkyPNG::Color.g(img[j, i])
        blue_array[i] << ChunkyPNG::Color.b(img[j, i])
      end
    end
    [red_array, green_array, blue_array]
  end
end
