module {
  func.func @main(%arg0: tensor<61x77xf32>) -> tensor<61x77xf32> {
    %0 = tosa.tanh %arg0 : (tensor<61x77xf32>) -> tensor<61x77xf32>
    return %0 : tensor<61x77xf32>
  }
}
