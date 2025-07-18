module {
  func.func @main(%arg0: tensor<7x60xi1>, %arg1: tensor<7x1xi1>, %arg2: tensor<43x19x30x32xi64>, %arg3: tensor<1x1x30x32xi64>) -> (tensor<7x60xi1>, tensor<43x19x30x32xi64>) {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<7x60xi1>, tensor<7x1xi1>) -> tensor<7x60xi1>
    %1 = tosa.minimum %arg2, %arg3 : (tensor<43x19x30x32xi64>, tensor<1x1x30x32xi64>) -> tensor<43x19x30x32xi64>
    return %0, %1 : tensor<7x60xi1>, tensor<43x19x30x32xi64>
  }
}
