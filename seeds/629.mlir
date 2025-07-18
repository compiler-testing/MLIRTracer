module {
  func.func @main(%arg0: tensor<17x69x66xi32>, %arg1: tensor<17x1x1xi32>) -> tensor<17x69x66xi32> {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<17x69x66xi32>, tensor<17x1x1xi32>) -> tensor<17x69x66xi32>
    return %0 : tensor<17x69x66xi32>
  }
}
