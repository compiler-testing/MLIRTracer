module {
  func.func @main(%arg0: tensor<1x46xi64>, %arg1: tensor<1x1xi64>, %arg2: tensor<65x41x33x18xf32>) -> (tensor<65x41x1x18xf32>, tensor<1x46xi1>, tensor<1x46xi1>, tensor<1x46xi1>) {
    %0 = tosa.equal %arg0, %arg1 : (tensor<1x46xi64>, tensor<1x1xi64>) -> tensor<1x46xi1>
    %1 = tosa.reciprocal %arg2 : (tensor<65x41x33x18xf32>) -> tensor<65x41x33x18xf32>
    %2 = tosa.bitwise_xor %0, %0 : (tensor<1x46xi1>, tensor<1x46xi1>) -> tensor<1x46xi1>
    %3 = tosa.identity %1 : (tensor<65x41x33x18xf32>) -> tensor<65x41x33x18xf32>
    %4 = tosa.arithmetic_right_shift %2, %0 {round = true} : (tensor<1x46xi1>, tensor<1x46xi1>) -> tensor<1x46xi1>
    %5 = tosa.reduce_product %3 {axis = 2 : i32} : (tensor<65x41x33x18xf32>) -> tensor<65x41x1x18xf32>
    %6 = tosa.maximum %5, %5 : (tensor<65x41x1x18xf32>, tensor<65x41x1x18xf32>) -> tensor<65x41x1x18xf32>
    %7 = tosa.arithmetic_right_shift %2, %0 {round = true} : (tensor<1x46xi1>, tensor<1x46xi1>) -> tensor<1x46xi1>
    %8 = tosa.logical_left_shift %7, %4 : (tensor<1x46xi1>, tensor<1x46xi1>) -> tensor<1x46xi1>
    %9 = tosa.logical_left_shift %2, %0 : (tensor<1x46xi1>, tensor<1x46xi1>) -> tensor<1x46xi1>
    %10 = tosa.arithmetic_right_shift %0, %4 {round = true} : (tensor<1x46xi1>, tensor<1x46xi1>) -> tensor<1x46xi1>
    return %6, %8, %9, %10 : tensor<65x41x1x18xf32>, tensor<1x46xi1>, tensor<1x46xi1>, tensor<1x46xi1>
  }
}
