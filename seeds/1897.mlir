module {
  func.func @main(%arg0: tensor<50x4x57x66x91xi64>, %arg1: tensor<50x4x1x1x91xi64>) -> tensor<50x4x57x66x91xi64> {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<50x4x57x66x91xi64>, tensor<50x4x1x1x91xi64>) -> tensor<50x4x57x66x91xi64>
    return %0 : tensor<50x4x57x66x91xi64>
  }
}
