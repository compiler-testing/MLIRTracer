module {
  func.func @main(%arg0: tensor<93xf32>, %arg1: tensor<i1>) -> (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<i1>, tensor<i1>) {
    %0 = tosa.reduce_product %arg0 {axis = 0 : i32} : (tensor<93xf32>) -> tensor<1xf32>
    %1 = tosa.reciprocal %0 : (tensor<1xf32>) -> tensor<1xf32>
    %2 = tosa.logical_not %arg1 : (tensor<i1>) -> tensor<i1>
    %3 = tosa.identity %2 : (tensor<i1>) -> tensor<i1>
    %4 = tosa.logical_and %3, %2 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %in_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %5 = tosa.negate %3, %in_zp_5, %out_zp_5 : (tensor<i1>, tensor<1xi1>, tensor<1xi1>) -> tensor<i1>
    %6 = tosa.bitwise_or %4, %3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %7 = tosa.reverse %1 {axis = 0 : i32} : (tensor<1xf32>) -> tensor<1xf32>
    %8 = tosa.pow %7, %1 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %9 = tosa.minimum %1, %7 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %10 = tosa.reduce_max %0 {axis = 0 : i32} : (tensor<1xf32>) -> tensor<1xf32>
    %in_zp_11 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_11 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %11 = tosa.negate %5, %in_zp_11, %out_zp_11 : (tensor<i1>, tensor<1xi1>, tensor<1xi1>) -> tensor<i1>
    %12 = tosa.bitwise_xor %6, %2 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    return %8, %9, %10, %11, %12 : tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<i1>, tensor<i1>
  }
}
