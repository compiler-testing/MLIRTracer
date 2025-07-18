module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<5x87xi8>) -> (tensor<f32>, tensor<11x1xi1>, tensor<10xi1>, tensor<1x87xi1>) {
    %0 = tosa.floor %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.reduce_max %arg1 {axis = 0 : i32} : (tensor<5x87xi8>) -> tensor<1x87xi8>
    %s_2_start = tosa.const_shape {values = dense<[ 0, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_2_size = tosa.const_shape {values = dense<[ 11, 8 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %2 = tosa.slice %1, %s_2_start, %s_2_size : (tensor<1x87xi8>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<11x8xi8>
    %3 = tosa.reciprocal %0 : (tensor<f32>) -> tensor<f32>
    %4 = tosa.greater_equal %2, %2 : (tensor<11x8xi8>, tensor<11x8xi8>) -> tensor<11x8xi1>
    %in_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %5 = tosa.negate %4, %in_zp_5, %out_zp_5 : (tensor<11x8xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<11x8xi1>
    %6 = tosa.reduce_all %5 {axis = 1 : i32} : (tensor<11x8xi1>) -> tensor<11x1xi1>
    %7 = tosa.clz %6 : (tensor<11x1xi1>) -> tensor<11x1xi1>
    %8 = tosa.greater %1, %1 : (tensor<1x87xi8>, tensor<1x87xi8>) -> tensor<1x87xi1>
    %s_9_start = tosa.const_shape {values = dense<[ 0, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_9_size = tosa.const_shape {values = dense<[ 10, 6 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %9 = tosa.slice %8, %s_9_start, %s_9_size : (tensor<1x87xi1>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<10x6xi1>
    %10 = tosa.logical_xor %9, %9 : (tensor<10x6xi1>, tensor<10x6xi1>) -> tensor<10x6xi1>
    %11 = tosa.reduce_all %10 {axis = 1 : i32} : (tensor<10x6xi1>) -> tensor<10x1xi1>
    %12 = tosa.logical_not %11 : (tensor<10x1xi1>) -> tensor<10x1xi1>
    %r_13 = tosa.const_shape {values = dense<[ 10 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %13 = tosa.reshape %12, %r_13 : (tensor<10x1xi1>, !tosa.shape<1>) -> tensor<10xi1>
    %14 = tosa.greater_equal %1, %1 : (tensor<1x87xi8>, tensor<1x87xi8>) -> tensor<1x87xi1>
    return %3, %7, %13, %14 : tensor<f32>, tensor<11x1xi1>, tensor<10xi1>, tensor<1x87xi1>
  }
}
