module {
  func.func @main(%arg0: tensor<75x3x49x48x20xi1>, %arg1: tensor<27xf32>, %arg2: tensor<27xf32>, %arg3: tensor<82x73xi1>) -> (tensor<82x1xi1>, tensor<1x1x10584000xi1>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) {
    %0 = tosa.logical_not %arg0 : (tensor<75x3x49x48x20xi1>) -> tensor<75x3x49x48x20xi1>
    %1 = tosa.logical_left_shift %0, %0 : (tensor<75x3x49x48x20xi1>, tensor<75x3x49x48x20xi1>) -> tensor<75x3x49x48x20xi1>
    %in_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %2 = tosa.negate %1, %in_zp_2, %out_zp_2 : (tensor<75x3x49x48x20xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<75x3x49x48x20xi1>
    %3 = tosa.pow %arg1, %arg2 : (tensor<27xf32>, tensor<27xf32>) -> tensor<27xf32>
    %4 = tosa.reduce_all %arg3 {axis = 1 : i32} : (tensor<82x73xi1>) -> tensor<82x1xi1>
    %r_5 = tosa.const_shape {values = dense<[ 1, 1, 10584000 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %5 = tosa.reshape %2, %r_5 : (tensor<75x3x49x48x20xi1>, !tosa.shape<3>) -> tensor<1x1x10584000xi1>
    %6 = tosa.reduce_sum %3 {axis = 0 : i32} : (tensor<27xf32>) -> tensor<1xf32>
    %7 = tosa.maximum %6, %6 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %8 = tosa.reduce_max %7 {axis = 0 : i32} : (tensor<1xf32>) -> tensor<1xf32>
    %in_zp_9 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_9 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %9 = tosa.negate %6, %in_zp_9, %out_zp_9 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %10 = tosa.reverse %6 {axis = 0 : i32} : (tensor<1xf32>) -> tensor<1xf32>
    return %4, %5, %8, %9, %10 : tensor<82x1xi1>, tensor<1x1x10584000xi1>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>
  }
}
