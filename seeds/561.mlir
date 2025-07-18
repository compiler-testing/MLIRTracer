module {
  func.func @main(%arg0: tensor<9x92x87x77x12xi64>, %arg1: tensor<1x92x1x1x1xi64>) -> tensor<9x92x87x77x12xi64> {
    %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<9x92x87x77x12xi64>, tensor<1x92x1x1x1xi64>) -> tensor<9x92x87x77x12xi64>
    return %0 : tensor<9x92x87x77x12xi64>
  }
}
