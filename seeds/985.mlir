module {
  func.func @main(%arg0: tensor<54x34x92x92xi64>, %arg1: tensor<54x34x92x1xi64>, %arg2: tensor<23x13x40x62x98xf32>) -> (tensor<612x25392xi64>, tensor<23x13x40x62x98xf32>, tensor<1x6xi1>) {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<54x34x92x92xi64>, tensor<54x34x92x1xi64>) -> tensor<54x34x92x92xi64>
    %1 = tosa.bitwise_and %0, %0 : (tensor<54x34x92x92xi64>, tensor<54x34x92x92xi64>) -> tensor<54x34x92x92xi64>
    %r_2 = tosa.const_shape {values = dense<[ 612, 25392 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %2 = tosa.reshape %1, %r_2 : (tensor<54x34x92x92xi64>, !tosa.shape<2>) -> tensor<612x25392xi64>
    %3 = tosa.greater_equal %2, %2 : (tensor<612x25392xi64>, tensor<612x25392xi64>) -> tensor<612x25392xi1>
    %4 = tosa.abs %3 : (tensor<612x25392xi1>) -> tensor<612x25392xi1>
    %5 = tosa.maximum %2, %2 : (tensor<612x25392xi64>, tensor<612x25392xi64>) -> tensor<612x25392xi64>
    %6 = tosa.log %arg2 : (tensor<23x13x40x62x98xf32>) -> tensor<23x13x40x62x98xf32>
    %s_7_start = tosa.const_shape {values = dense<[ 413, 453 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_7_size = tosa.const_shape {values = dense<[ 5, 6 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %7 = tosa.slice %4, %s_7_start, %s_7_size : (tensor<612x25392xi1>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<5x6xi1>
    %8 = tosa.bitwise_and %7, %7 : (tensor<5x6xi1>, tensor<5x6xi1>) -> tensor<5x6xi1>
    %in_zp_9 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_9 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %9 = tosa.negate %8, %in_zp_9, %out_zp_9 : (tensor<5x6xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<5x6xi1>
    %10 = tosa.reduce_max %9 {axis = 0 : i32} : (tensor<5x6xi1>) -> tensor<1x6xi1>
    return %5, %6, %10 : tensor<612x25392xi64>, tensor<23x13x40x62x98xf32>, tensor<1x6xi1>
  }
}
