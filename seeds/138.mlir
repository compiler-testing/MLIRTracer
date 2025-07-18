module {
  func.func @main(%arg0: tensor<85x83x17x62xi32>, %arg1: tensor<85x1x1x1xi32>, %arg2: tensor<48x6xf32>) -> (tensor<48x6xf32>, tensor<1x83x17x62xi32>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<85x83x17x62xi32>, tensor<85x1x1x1xi32>) -> tensor<85x83x17x62xi32>
    %1 = tosa.reduce_sum %0 {axis = 0 : i32} : (tensor<85x83x17x62xi32>) -> tensor<1x83x17x62xi32>
    %2 = tosa.intdiv %1, %1 : (tensor<1x83x17x62xi32>, tensor<1x83x17x62xi32>) -> tensor<1x83x17x62xi32>
    %3 = tosa.exp %arg2 : (tensor<48x6xf32>) -> tensor<48x6xf32>
    %4 = tosa.arithmetic_right_shift %2, %1 {round = true} : (tensor<1x83x17x62xi32>, tensor<1x83x17x62xi32>) -> tensor<1x83x17x62xi32>
    %5 = tosa.clz %4 : (tensor<1x83x17x62xi32>) -> tensor<1x83x17x62xi32>
    return %3, %5 : tensor<48x6xf32>, tensor<1x83x17x62xi32>
  }
}
