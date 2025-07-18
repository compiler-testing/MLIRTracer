module {
  func.func @main(%arg0: tensor<7xi32>, %arg1: tensor<1xi32>, %arg2: tensor<44x20xi1>, %arg3: tensor<1x1xi1>, %arg4: tensor<16x67x18xf32>) -> (tensor<7xi32>, tensor<44x20xi1>, tensor<16x67x18xf32>) {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = true} : (tensor<7xi32>, tensor<1xi32>) -> tensor<7xi32>
    %1 = tosa.logical_xor %arg2, %arg3 : (tensor<44x20xi1>, tensor<1x1xi1>) -> tensor<44x20xi1>
    %2 = tosa.logical_right_shift %1, %1 : (tensor<44x20xi1>, tensor<44x20xi1>) -> tensor<44x20xi1>
    %3 = tosa.ceil %arg4 : (tensor<16x67x18xf32>) -> tensor<16x67x18xf32>
    return %0, %2, %3 : tensor<7xi32>, tensor<44x20xi1>, tensor<16x67x18xf32>
  }
}
