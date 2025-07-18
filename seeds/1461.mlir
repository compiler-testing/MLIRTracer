module {
  func.func @main(%arg0: tensor<48xf32>) -> tensor<48xf32> {
    %0 = tosa.floor %arg0 : (tensor<48xf32>) -> tensor<48xf32>
    return %0 : tensor<48xf32>
  }
}
