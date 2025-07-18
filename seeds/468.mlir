module {
  func.func @main(%arg0: tensor<66xf32>, %arg1: tensor<1xf32>, %arg2: tensor<76x14x9x62xi1>, %arg3: tensor<1x14x1x1xi1>) -> (tensor<1xf32>, tensor<66xf32>, tensor<132xf32>, tensor<76x14x1x62xi1>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<66xf32>, tensor<1xf32>) -> tensor<66xf32>
    %1 = tosa.logical_or %arg2, %arg3 : (tensor<76x14x9x62xi1>, tensor<1x14x1x1xi1>) -> tensor<76x14x9x62xi1>
    %in_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.negate %0, %in_zp_2, %out_zp_2 : (tensor<66xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<66xf32>
    %t_3 = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %3 = tosa.tile %2, %t_3 : (tensor<66xf32>, !tosa.shape<1>) -> tensor<132xf32>
    %s_4_start = tosa.const_shape {values = dense<[ 32 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_4_size = tosa.const_shape {values = dense<[ 9 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %4 = tosa.slice %3, %s_4_start, %s_4_size : (tensor<132xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<9xf32>
    %5 = tosa.reduce_max %4 {axis = 0 : i32} : (tensor<9xf32>) -> tensor<1xf32>
    %6 = tosa.bitwise_xor %1, %1 : (tensor<76x14x9x62xi1>, tensor<76x14x9x62xi1>) -> tensor<76x14x9x62xi1>
    %7 = tosa.floor %2 : (tensor<66xf32>) -> tensor<66xf32>
    %8 = tosa.exp %3 : (tensor<132xf32>) -> tensor<132xf32>
    %9 = tosa.reduce_product %6 {axis = 2 : i32} : (tensor<76x14x9x62xi1>) -> tensor<76x14x1x62xi1>
    return %5, %7, %8, %9 : tensor<1xf32>, tensor<66xf32>, tensor<132xf32>, tensor<76x14x1x62xi1>
  }
}
