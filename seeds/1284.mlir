module {
  func.func @main(%arg0: tensor<65x70x78x13xi1>, %arg1: tensor<65x1x78x13xi1>) -> tensor<65x70x78x13xi1> {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<65x70x78x13xi1>, tensor<65x1x78x13xi1>) -> tensor<65x70x78x13xi1>
    return %0 : tensor<65x70x78x13xi1>
  }
}
