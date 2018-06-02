class Network
  protected

  def save_data(path, array)
    serialized_array = Marshal.dump(array)
    File.open(path, 'wb') { |f| f.write(serialized_array) }
    File.open(path + '.sha512', 'w') { |f| f.write(Digest::SHA512.file(path)) }
  end

  def load_data(key, path)
    tmp = nil
    if File.read(key) == Digest::SHA512.file(path).to_s
      tmp = Marshal.load File.open(path, 'rb')
    else
      puts 'SHA512 sum does not match'
    end
    tmp
  end
end