module {
  func.func @main(%arg0: tensor<96x99xi32>, %arg1: tensor<96x99xi32>, %arg2: tensor<93x65x87x100x42xf32>) -> (tensor<93x65x87x100x42xf32>, tensor<93x65x87x100x42xf32>, tensor<96x99xi1>, tensor<96x99xi1>, tensor<1x99xi1>) {
    %0 = tosa.equal %arg0, %arg1 : (tensor<96x99xi32>, tensor<96x99xi32>) -> tensor<96x99xi1>
    %1 = tosa.bitwise_xor %0, %0 : (tensor<96x99xi1>, tensor<96x99xi1>) -> tensor<96x99xi1>
    %2 = tosa.logical_left_shift %1, %0 : (tensor<96x99xi1>, tensor<96x99xi1>) -> tensor<96x99xi1>
    %3 = tosa.floor %arg2 : (tensor<93x65x87x100x42xf32>) -> tensor<93x65x87x100x42xf32>
    %4 = tosa.tanh %3 : (tensor<93x65x87x100x42xf32>) -> tensor<93x65x87x100x42xf32>
    %t_5 = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %5 = tosa.tile %0, %t_5 : (tensor<96x99xi1>, !tosa.shape<2>) -> tensor<96x99xi1>
    %in_zp_6 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_6 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %6 = tosa.negate %3, %in_zp_6, %out_zp_6 : (tensor<93x65x87x100x42xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<93x65x87x100x42xf32>
    %7 = tosa.tanh %6 : (tensor<93x65x87x100x42xf32>) -> tensor<93x65x87x100x42xf32>
    %8 = tosa.bitwise_or %0, %1 : (tensor<96x99xi1>, tensor<96x99xi1>) -> tensor<96x99xi1>
    %9 = tosa.logical_and %5, %2 : (tensor<96x99xi1>, tensor<96x99xi1>) -> tensor<96x99xi1>
    %10 = tosa.reduce_max %1 {axis = 0 : i32} : (tensor<96x99xi1>) -> tensor<1x99xi1>
    return %4, %7, %8, %9, %10 : tensor<93x65x87x100x42xf32>, tensor<93x65x87x100x42xf32>, tensor<96x99xi1>, tensor<96x99xi1>, tensor<1x99xi1>
  }
}
