module {
  func.func @main(%arg0: tensor<17x57x1xi32>, %arg1: tensor<17x1x75xi32>, %arg2: tensor<58xi1>, %arg3: tensor<58xi1>, %arg4: tensor<17x77xf32>) -> (tensor<17x57x75xi32>, tensor<17x57x75xi1>, tensor<58xi1>, tensor<1x77xf32>, tensor<1xi1>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<17x57x1xi32>, tensor<17x1x75xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<17x57x75xi32>
    %1 = tosa.minimum %0, %0 : (tensor<17x57x75xi32>, tensor<17x57x75xi32>) -> tensor<17x57x75xi32>
    %2 = tosa.arithmetic_right_shift %1, %0 {round = false} : (tensor<17x57x75xi32>, tensor<17x57x75xi32>) -> tensor<17x57x75xi32>
    %3 = tosa.logical_xor %arg2, %arg3 : (tensor<58xi1>, tensor<58xi1>) -> tensor<58xi1>
    %4 = tosa.logical_and %3, %3 : (tensor<58xi1>, tensor<58xi1>) -> tensor<58xi1>
    %5 = tosa.greater_equal %0, %0 : (tensor<17x57x75xi32>, tensor<17x57x75xi32>) -> tensor<17x57x75xi1>
    %6 = tosa.logical_not %4 : (tensor<58xi1>) -> tensor<58xi1>
    %7 = tosa.log %arg4 : (tensor<17x77xf32>) -> tensor<17x77xf32>
    %8 = tosa.bitwise_and %3, %3 : (tensor<58xi1>, tensor<58xi1>) -> tensor<58xi1>
    %9 = tosa.reverse %6 {axis = 0 : i32} : (tensor<58xi1>) -> tensor<58xi1>
    %10 = tosa.bitwise_not %9 : (tensor<58xi1>) -> tensor<58xi1>
    %11 = tosa.logical_xor %8, %6 : (tensor<58xi1>, tensor<58xi1>) -> tensor<58xi1>
    %12 = tosa.reduce_sum %7 {axis = 0 : i32} : (tensor<17x77xf32>) -> tensor<1x77xf32>
    %13 = tosa.reduce_any %10 {axis = 0 : i32} : (tensor<58xi1>) -> tensor<1xi1>
    return %2, %5, %11, %12, %13 : tensor<17x57x75xi32>, tensor<17x57x75xi1>, tensor<58xi1>, tensor<1x77xf32>, tensor<1xi1>
  }
}
