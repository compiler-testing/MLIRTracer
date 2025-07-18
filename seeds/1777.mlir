module {
  func.func @main(%arg0: tensor<65x55x24xi32>, %arg1: tensor<65x1x1xi32>) -> tensor<65x55x24xi32> {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<65x55x24xi32>, tensor<65x1x1xi32>) -> tensor<65x55x24xi32>
    %1 = tosa.bitwise_not %0 : (tensor<65x55x24xi32>) -> tensor<65x55x24xi32>
    return %1 : tensor<65x55x24xi32>
  }
}
