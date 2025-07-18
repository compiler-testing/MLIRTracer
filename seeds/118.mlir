module {
  func.func @main(%arg0: tensor<11x94x65xf32>) -> tensor<11x94x65xf32> {
    %0 = tosa.sigmoid %arg0 : (tensor<11x94x65xf32>) -> tensor<11x94x65xf32>
    return %0 : tensor<11x94x65xf32>
  }
}
