module {
  func.func @main(%arg0: tensor<23x13xi16>, %arg1: tensor<1x13xi16>) -> tensor<23x13xi16> {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<23x13xi16>, tensor<1x13xi16>) -> tensor<23x13xi16>
    return %0 : tensor<23x13xi16>
  }
}
