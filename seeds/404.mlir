module {
  func.func @main(%arg0: tensor<56x40x78x96x1x74xf32>, %arg1: tensor<83x12x86x3xi16>, %arg2: tensor<1x1x86x3xi16>) -> (tensor<56x40x78x96x1x74xf32>, tensor<83x12x86x3xi16>) {
    %0 = tosa.sigmoid %arg0 : (tensor<56x40x78x96x1x74xf32>) -> tensor<56x40x78x96x1x74xf32>
    %1 = tosa.arithmetic_right_shift %arg1, %arg2 {round = true} : (tensor<83x12x86x3xi16>, tensor<1x1x86x3xi16>) -> tensor<83x12x86x3xi16>
    %2 = tosa.identity %1 : (tensor<83x12x86x3xi16>) -> tensor<83x12x86x3xi16>
    return %0, %2 : tensor<56x40x78x96x1x74xf32>, tensor<83x12x86x3xi16>
  }
}
