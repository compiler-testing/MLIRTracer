module {
  func.func @main(%arg0: tensor<54x19x81xi8>, %arg1: tensor<54x19x1xi8>, %arg2: tensor<74x91x8x84x39x92xf32>) -> (tensor<1x19x81xi1>, tensor<74x91x8x84x39x92xf32>, tensor<1x19x1xi1>, tensor<74x91x8x84x39x184xi1>) {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<54x19x81xi8>, tensor<54x19x1xi8>) -> tensor<54x19x81xi1>
    %1 = tosa.reciprocal %arg2 : (tensor<74x91x8x84x39x92xf32>) -> tensor<74x91x8x84x39x92xf32>
    %2 = tosa.reduce_product %0 {axis = 0 : i32} : (tensor<54x19x81xi1>) -> tensor<1x19x81xi1>
    %3 = tosa.sub %1, %1 : (tensor<74x91x8x84x39x92xf32>, tensor<74x91x8x84x39x92xf32>) -> tensor<74x91x8x84x39x92xf32>
    %4 = tosa.rsqrt %3 : (tensor<74x91x8x84x39x92xf32>) -> tensor<74x91x8x84x39x92xf32>
    %5 = tosa.reduce_any %2 {axis = 0 : i32} : (tensor<1x19x81xi1>) -> tensor<1x19x81xi1>
    %6 = tosa.reduce_product %5 {axis = 0 : i32} : (tensor<1x19x81xi1>) -> tensor<1x19x81xi1>
    %7 = tosa.concat %3, %4 {axis = 5 : i32} : (tensor<74x91x8x84x39x92xf32>, tensor<74x91x8x84x39x92xf32>) -> tensor<74x91x8x84x39x184xf32>
    %in_zp_8 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_8 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %8 = tosa.negate %4, %in_zp_8, %out_zp_8 : (tensor<74x91x8x84x39x92xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<74x91x8x84x39x92xf32>
    %9 = tosa.reduce_max %0 {axis = 0 : i32} : (tensor<54x19x81xi1>) -> tensor<1x19x81xi1>
    %10 = tosa.reciprocal %8 : (tensor<74x91x8x84x39x92xf32>) -> tensor<74x91x8x84x39x92xf32>
    %11 = tosa.reduce_sum %6 {axis = 2 : i32} : (tensor<1x19x81xi1>) -> tensor<1x19x1xi1>
    %12 = tosa.equal %7, %7 : (tensor<74x91x8x84x39x184xf32>, tensor<74x91x8x84x39x184xf32>) -> tensor<74x91x8x84x39x184xi1>
    return %9, %10, %11, %12 : tensor<1x19x81xi1>, tensor<74x91x8x84x39x92xf32>, tensor<1x19x1xi1>, tensor<74x91x8x84x39x184xi1>
  }
}
