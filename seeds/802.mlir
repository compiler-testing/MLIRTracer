module {
  func.func @main(%arg0: tensor<36xf32>) -> tensor<36xf32> {
    %0 = tosa.ceil %arg0 : (tensor<36xf32>) -> tensor<36xf32>
    return %0 : tensor<36xf32>
  }
}
