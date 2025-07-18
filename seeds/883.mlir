module {
  func.func @main(%arg0: tensor<88x32xf32>) -> tensor<88x32xf32> {
    %0 = tosa.tanh %arg0 : (tensor<88x32xf32>) -> tensor<88x32xf32>
    return %0 : tensor<88x32xf32>
  }
}
