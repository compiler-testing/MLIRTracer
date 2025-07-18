module {
  func.func @main(%arg0: tensor<25x64x68xi64>, %arg1: tensor<1x64x1xi64>) -> tensor<25x64x68xi64> {
    %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<25x64x68xi64>, tensor<1x64x1xi64>) -> tensor<25x64x68xi64>
    return %0 : tensor<25x64x68xi64>
  }
}
