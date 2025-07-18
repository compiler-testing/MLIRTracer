module {
  func.func @main(%arg0: tensor<28x8x79x77x86x38xf32>) -> tensor<28x8x79x77x86x38xf32> {
    %0 = tosa.exp %arg0 : (tensor<28x8x79x77x86x38xf32>) -> tensor<28x8x79x77x86x38xf32>
    return %0 : tensor<28x8x79x77x86x38xf32>
  }
}
