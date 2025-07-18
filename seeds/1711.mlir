module {
  func.func @main(%arg0: tensor<37xf32>) -> tensor<37xf32> {
    %0 = tosa.tanh %arg0 : (tensor<37xf32>) -> tensor<37xf32>
    return %0 : tensor<37xf32>
  }
}
