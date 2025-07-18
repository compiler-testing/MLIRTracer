module {
  func.func @main(%arg0: tensor<60x76x92xf32>, %arg1: tensor<60x92x31xf32>, %arg2: tensor<90x61x20x62xi1>, %arg3: tensor<90x61x20x62xi1>) -> (tensor<90x61x20x62xi1>, tensor<60x76x31xf32>, tensor<90x1x1x62xi1>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %b_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<60x76x92xf32>, tensor<60x92x31xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<60x76x31xf32>
    %1 = tosa.logical_right_shift %arg2, %arg3 : (tensor<90x61x20x62xi1>, tensor<90x61x20x62xi1>) -> tensor<90x61x20x62xi1>
    %2 = tosa.bitwise_xor %1, %1 : (tensor<90x61x20x62xi1>, tensor<90x61x20x62xi1>) -> tensor<90x61x20x62xi1>
    %3 = tosa.reduce_max %1 {axis = 1 : i32} : (tensor<90x61x20x62xi1>) -> tensor<90x1x20x62xi1>
    %4 = tosa.sub %2, %1 : (tensor<90x61x20x62xi1>, tensor<90x61x20x62xi1>) -> tensor<90x61x20x62xi1>
    %in_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %5 = tosa.negate %4, %in_zp_5, %out_zp_5 : (tensor<90x61x20x62xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<90x61x20x62xi1>
    %6 = tosa.rsqrt %0 : (tensor<60x76x31xf32>) -> tensor<60x76x31xf32>
    %7 = tosa.logical_or %3, %3 : (tensor<90x1x20x62xi1>, tensor<90x1x20x62xi1>) -> tensor<90x1x20x62xi1>
    %8 = tosa.reciprocal %6 : (tensor<60x76x31xf32>) -> tensor<60x76x31xf32>
    %9 = tosa.reduce_all %7 {axis = 2 : i32} : (tensor<90x1x20x62xi1>) -> tensor<90x1x1x62xi1>
    return %5, %8, %9 : tensor<90x61x20x62xi1>, tensor<60x76x31xf32>, tensor<90x1x1x62xi1>
  }
}
