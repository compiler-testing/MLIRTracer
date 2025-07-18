module {
  func.func @main(%arg0: tensor<57x84xf32>) -> tensor<57x84xf32> {
    %0 = tosa.tanh %arg0 : (tensor<57x84xf32>) -> tensor<57x84xf32>
    return %0 : tensor<57x84xf32>
  }
}
