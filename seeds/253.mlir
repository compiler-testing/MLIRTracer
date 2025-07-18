module {
  func.func @main(%arg0: tensor<28x80x95xi1>, %arg1: tensor<9xf32>) -> (tensor<28x80x95xi1>, tensor<9xf32>) {
    %0 = tosa.logical_not %arg0 : (tensor<28x80x95xi1>) -> tensor<28x80x95xi1>
    %1 = tosa.log %arg1 : (tensor<9xf32>) -> tensor<9xf32>
    return %0, %1 : tensor<28x80x95xi1>, tensor<9xf32>
  }
}
