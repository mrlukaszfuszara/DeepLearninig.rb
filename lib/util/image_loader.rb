require 'chunky_png'

class ImageLoader
  def load_image(image_path)
    img = ChunkyPNG::Image.from_file(image_path)

    height = img.dimension.height
    width  = img.dimension.width

    array_of_pixels = []

    height.times do |i|
      array_of_pixels[i] = []
      width.times do |j|
        array_of_pixels[i] << [ChunkyPNG::Color.r(img[j, i]), ChunkyPNG::Color.g(img[j, i]), ChunkyPNG::Color.b(img[j, i])]
      end
    end
    array_of_pixels
  end
end
