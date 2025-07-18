module {
  func.func @main(%arg0: tensor<88xf32>) -> tensor<88xf32> {
    %0 = tosa.tanh %arg0 : (tensor<88xf32>) -> tensor<88xf32>
    %1 = tosa.identity %0 : (tensor<88xf32>) -> tensor<88xf32>
    return %1 : tensor<88xf32>
  }
}
