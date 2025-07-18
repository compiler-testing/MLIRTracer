module {
  func.func @main(%arg0: tensor<58x92xi1>, %arg1: tensor<1x1xi1>) -> tensor<58x92xi1> {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<58x92xi1>, tensor<1x1xi1>) -> tensor<58x92xi1>
    %1 = tosa.logical_xor %0, %0 : (tensor<58x92xi1>, tensor<58x92xi1>) -> tensor<58x92xi1>
    return %1 : tensor<58x92xi1>
  }
}
