module {
  func.func @main(%arg0: tensor<72x41x98x70xf32>) -> tensor<72x41x98x70xf32> {
    %0 = tosa.tanh %arg0 : (tensor<72x41x98x70xf32>) -> tensor<72x41x98x70xf32>
    return %0 : tensor<72x41x98x70xf32>
  }
}
