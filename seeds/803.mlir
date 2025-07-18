module {
  func.func @main(%arg0: tensor<41x74x41x32xi32>, %arg1: tensor<41x74x1x1xi32>, %arg2: tensor<f32>) -> (tensor<f32>, tensor<1x1x41x32xi1>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<41x74x41x32xi32>, tensor<41x74x1x1xi32>) -> tensor<41x74x41x32xi32>
    %1 = tosa.reduce_product %0 {axis = 1 : i32} : (tensor<41x74x41x32xi32>) -> tensor<41x1x41x32xi32>
    %2 = tosa.reciprocal %arg2 : (tensor<f32>) -> tensor<f32>
    %3 = tosa.arithmetic_right_shift %1, %1 {round = true} : (tensor<41x1x41x32xi32>, tensor<41x1x41x32xi32>) -> tensor<41x1x41x32xi32>
    %4 = tosa.reduce_max %3 {axis = 1 : i32} : (tensor<41x1x41x32xi32>) -> tensor<41x1x41x32xi32>
    %5 = tosa.identity %4 : (tensor<41x1x41x32xi32>) -> tensor<41x1x41x32xi32>
    %6 = tosa.bitwise_xor %5, %5 : (tensor<41x1x41x32xi32>, tensor<41x1x41x32xi32>) -> tensor<41x1x41x32xi32>
    %7 = tosa.concat %6, %1 {axis = 0 : i32} : (tensor<41x1x41x32xi32>, tensor<41x1x41x32xi32>) -> tensor<82x1x41x32xi32>
    %8 = tosa.intdiv %7, %7 : (tensor<82x1x41x32xi32>, tensor<82x1x41x32xi32>) -> tensor<82x1x41x32xi32>
    %9 = tosa.minimum %8, %8 : (tensor<82x1x41x32xi32>, tensor<82x1x41x32xi32>) -> tensor<82x1x41x32xi32>
    %10 = tosa.reduce_sum %9 {axis = 0 : i32} : (tensor<82x1x41x32xi32>) -> tensor<1x1x41x32xi32>
    %11 = tosa.intdiv %10, %10 : (tensor<1x1x41x32xi32>, tensor<1x1x41x32xi32>) -> tensor<1x1x41x32xi32>
    %12 = tosa.greater %11, %11 : (tensor<1x1x41x32xi32>, tensor<1x1x41x32xi32>) -> tensor<1x1x41x32xi1>
    return %2, %12 : tensor<f32>, tensor<1x1x41x32xi1>
  }
}
