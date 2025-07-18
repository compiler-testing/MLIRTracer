module {
  func.func @main(%arg0: tensor<98x23x26xi1>, %arg1: tensor<98x1x1xi1>) -> tensor<98x23x26xi1> {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<98x23x26xi1>, tensor<98x1x1xi1>) -> tensor<98x23x26xi1>
    %1 = tosa.bitwise_xor %0, %0 : (tensor<98x23x26xi1>, tensor<98x23x26xi1>) -> tensor<98x23x26xi1>
    %2 = tosa.logical_or %1, %0 : (tensor<98x23x26xi1>, tensor<98x23x26xi1>) -> tensor<98x23x26xi1>
    return %2 : tensor<98x23x26xi1>
  }
}
