module {
  func.func @main(%arg0: tensor<86xi32>, %arg1: tensor<86xi32>) -> tensor<86xi32> {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<86xi32>, tensor<86xi32>) -> tensor<86xi32>
    %1 = tosa.bitwise_not %0 : (tensor<86xi32>) -> tensor<86xi32>
    %2 = tosa.bitwise_or %1, %1 : (tensor<86xi32>, tensor<86xi32>) -> tensor<86xi32>
    %3 = tosa.logical_left_shift %2, %2 : (tensor<86xi32>, tensor<86xi32>) -> tensor<86xi32>
    return %3 : tensor<86xi32>
  }
}
