module {
  func.func @main(%arg0: tensor<97x14x71x97x87x16xi32>, %arg1: tensor<1x1x1x97x1x1xi32>) -> tensor<97x14x71x97x87x16xi32> {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<97x14x71x97x87x16xi32>, tensor<1x1x1x97x1x1xi32>) -> tensor<97x14x71x97x87x16xi32>
    return %0 : tensor<97x14x71x97x87x16xi32>
  }
}
