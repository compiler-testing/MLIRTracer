module {
  func.func @main(%arg0: tensor<72x25x34x16xf32>) -> tensor<72x25x34x16xf32> {
    %0 = tosa.reciprocal %arg0 : (tensor<72x25x34x16xf32>) -> tensor<72x25x34x16xf32>
    return %0 : tensor<72x25x34x16xf32>
  }
}
