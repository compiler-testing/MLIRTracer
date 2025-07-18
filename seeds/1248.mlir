module {
  func.func @main(%arg0: tensor<8x69x87xi64>, %arg1: tensor<8x1x87xi64>) -> tensor<8x69x87xi64> {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<8x69x87xi64>, tensor<8x1x87xi64>) -> tensor<8x69x87xi64>
    return %0 : tensor<8x69x87xi64>
  }
}
