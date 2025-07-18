module {
  func.func @main(%arg0: tensor<94x77x34x90x8xf32>) -> tensor<94x77x34x90x8xf32> {
    %0 = tosa.floor %arg0 : (tensor<94x77x34x90x8xf32>) -> tensor<94x77x34x90x8xf32>
    return %0 : tensor<94x77x34x90x8xf32>
  }
}
