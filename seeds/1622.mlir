module {
  func.func @main(%arg0: tensor<43x13xi1>, %arg1: tensor<1x1xi1>) -> tensor<43x13xi1> {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<43x13xi1>, tensor<1x1xi1>) -> tensor<43x13xi1>
    return %0 : tensor<43x13xi1>
  }
}
