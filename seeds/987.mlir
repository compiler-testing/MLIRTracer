module {
  func.func @main(%arg0: tensor<53x41xf32>) -> tensor<53x41xf32> {
    %0 = tosa.floor %arg0 : (tensor<53x41xf32>) -> tensor<53x41xf32>
    return %0 : tensor<53x41xf32>
  }
}
