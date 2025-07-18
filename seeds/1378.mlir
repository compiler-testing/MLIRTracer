module {
  func.func @main(%arg0: tensor<65xf32>, %arg1: tensor<56x7x41xi1>, %arg2: tensor<44x46x73x82xi32>, %arg3: tensor<1x46x1x1xi32>) -> (tensor<1xf32>, tensor<44x46x73x82xi32>, tensor<56x41x7xi1>, tensor<1xf32>) {
    %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<65xf32>) -> tensor<1xf32>
    %1 = tosa.identity %0 : (tensor<1xf32>) -> tensor<1xf32>
    %in_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.negate %1, %in_zp_2, %out_zp_2 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %3 = tosa.logical_not %arg1 : (tensor<56x7x41xi1>) -> tensor<56x7x41xi1>
    %4 = "tosa.const"() {values = dense<[0, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
    %5 = tosa.transpose %3 {perms = array<i32: 0, 2, 1>} : (tensor<56x7x41xi1>) -> tensor<56x41x7xi1>
    %6 = tosa.bitwise_and %5, %5 : (tensor<56x41x7xi1>, tensor<56x41x7xi1>) -> tensor<56x41x7xi1>
    %7 = tosa.intdiv %arg2, %arg3 : (tensor<44x46x73x82xi32>, tensor<1x46x1x1xi32>) -> tensor<44x46x73x82xi32>
    %8 = tosa.sub %6, %6 : (tensor<56x41x7xi1>, tensor<56x41x7xi1>) -> tensor<56x41x7xi1>
    %9 = tosa.reverse %8 {axis = 2 : i32} : (tensor<56x41x7xi1>) -> tensor<56x41x7xi1>
    %10 = tosa.log %1 : (tensor<1xf32>) -> tensor<1xf32>
    %11 = tosa.sigmoid %10 : (tensor<1xf32>) -> tensor<1xf32>
    return %2, %7, %9, %11 : tensor<1xf32>, tensor<44x46x73x82xi32>, tensor<56x41x7xi1>, tensor<1xf32>
  }
}
