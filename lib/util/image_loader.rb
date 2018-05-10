require 'chunky_png'

class ImageLoader
  def load_image(image_path)
    array_of_pixels = []
    png_stream = ChunkyPNG::Datastream.from_file(image_path)
    png_stream.each_chunk { |chunk| p chunk.type }
  end
end

#il = ImageLoader.new
#il.load_image()