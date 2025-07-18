module {
  func.func @main(%arg0: tensor<31x36xf32>) -> tensor<31x36xf32> {
    %0 = tosa.tanh %arg0 : (tensor<31x36xf32>) -> tensor<31x36xf32>
    %1 = tosa.sigmoid %0 : (tensor<31x36xf32>) -> tensor<31x36xf32>
    return %1 : tensor<31x36xf32>
  }
}
