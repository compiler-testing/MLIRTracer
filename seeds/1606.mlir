module {
  func.func @main(%arg0: tensor<71x66x38x65x77x13xf32>, %arg1: tensor<95xi32>, %arg2: tensor<85x24xi1>) -> (tensor<i32>, tensor<71x66x38x65x77x13xf32>, tensor<1x24xi1>, tensor<24xi32>) {
    %0 = tosa.rsqrt %arg0 : (tensor<71x66x38x65x77x13xf32>) -> tensor<71x66x38x65x77x13xf32>
    %1 = tosa.argmax %arg1 {axis = 0 : i32} : (tensor<95xi32>) -> tensor<i32>
    %2 = tosa.reduce_any %arg2 {axis = 0 : i32} : (tensor<85x24xi1>) -> tensor<1x24xi1>
    %3 = tosa.argmax %2 {axis = 0 : i32} : (tensor<1x24xi1>) -> tensor<24xi32>
    %4 = tosa.floor %0 : (tensor<71x66x38x65x77x13xf32>) -> tensor<71x66x38x65x77x13xf32>
    %5 = tosa.reverse %3 {axis = 0 : i32} : (tensor<24xi32>) -> tensor<24xi32>
    %6 = tosa.logical_left_shift %5, %5 : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %7 = tosa.bitwise_or %2, %2 : (tensor<1x24xi1>, tensor<1x24xi1>) -> tensor<1x24xi1>
    %in_zp_8 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_8 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %8 = tosa.negate %6, %in_zp_8, %out_zp_8 : (tensor<24xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<24xi32>
    return %1, %4, %7, %8 : tensor<i32>, tensor<71x66x38x65x77x13xf32>, tensor<1x24xi1>, tensor<24xi32>
  }
}
