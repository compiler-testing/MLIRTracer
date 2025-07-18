module {
  func.func @main(%arg0: tensor<9x2x73x89xf32>) -> tensor<9x2x73x89xf32> {
    %0 = tosa.tanh %arg0 : (tensor<9x2x73x89xf32>) -> tensor<9x2x73x89xf32>
    return %0 : tensor<9x2x73x89xf32>
  }
}
