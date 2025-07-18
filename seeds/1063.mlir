module {
  func.func @main(%arg0: tensor<80xf32>, %arg1: tensor<80xf32>, %arg2: tensor<91x96x39xi1>, %arg3: tensor<1x96x39xi1>) -> (tensor<91x96x1xi1>, tensor<80xf32>, tensor<1x96x39xi1>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<80xf32>, tensor<80xf32>) -> tensor<80xf32>
    %1 = tosa.arithmetic_right_shift %arg2, %arg3 {round = true} : (tensor<91x96x39xi1>, tensor<1x96x39xi1>) -> tensor<91x96x39xi1>
    %2 = tosa.clz %1 : (tensor<91x96x39xi1>) -> tensor<91x96x39xi1>
    %3 = tosa.bitwise_xor %1, %2 : (tensor<91x96x39xi1>, tensor<91x96x39xi1>) -> tensor<91x96x39xi1>
    %4 = tosa.reduce_max %2 {axis = 2 : i32} : (tensor<91x96x39xi1>) -> tensor<91x96x1xi1>
    %5 = tosa.reciprocal %0 : (tensor<80xf32>) -> tensor<80xf32>
    %6 = tosa.reduce_all %3 {axis = 0 : i32} : (tensor<91x96x39xi1>) -> tensor<1x96x39xi1>
    %7 = tosa.maximum %0, %5 : (tensor<80xf32>, tensor<80xf32>) -> tensor<80xf32>
    %8 = tosa.arithmetic_right_shift %6, %6 {round = true} : (tensor<1x96x39xi1>, tensor<1x96x39xi1>) -> tensor<1x96x39xi1>
    return %4, %7, %8 : tensor<91x96x1xi1>, tensor<80xf32>, tensor<1x96x39xi1>
  }
}
