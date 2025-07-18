module {
  func.func @main(%arg0: tensor<100x52xf32>) -> (tensor<1x52xf32>, tensor<5x1xi1>) {
    %0 = tosa.tanh %arg0 : (tensor<100x52xf32>) -> tensor<100x52xf32>
    %1 = tosa.log %0 : (tensor<100x52xf32>) -> tensor<100x52xf32>
    %2 = tosa.reduce_min %1 {axis = 0 : i32} : (tensor<100x52xf32>) -> tensor<1x52xf32>
    %3 = tosa.sub %2, %2 : (tensor<1x52xf32>, tensor<1x52xf32>) -> tensor<1x52xf32>
    %4 = tosa.abs %3 : (tensor<1x52xf32>) -> tensor<1x52xf32>
    %5 = tosa.reverse %4 {axis = 0 : i32} : (tensor<1x52xf32>) -> tensor<1x52xf32>
    %6 = tosa.tanh %5 : (tensor<1x52xf32>) -> tensor<1x52xf32>
    %7 = tosa.greater %6, %4 : (tensor<1x52xf32>, tensor<1x52xf32>) -> tensor<1x52xi1>
    %8 = tosa.sigmoid %4 : (tensor<1x52xf32>) -> tensor<1x52xf32>
    %9 = tosa.reduce_product %7 {axis = 0 : i32} : (tensor<1x52xi1>) -> tensor<1x52xi1>
    %in_zp_10 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_10 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %10 = tosa.negate %9, %in_zp_10, %out_zp_10 : (tensor<1x52xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<1x52xi1>
    %11 = tosa.bitwise_not %10 : (tensor<1x52xi1>) -> tensor<1x52xi1>
    %s_12_start = tosa.const_shape {values = dense<[ 0, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_12_size = tosa.const_shape {values = dense<[ 5, 8 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %12 = tosa.slice %11, %s_12_start, %s_12_size : (tensor<1x52xi1>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<5x8xi1>
    %13 = tosa.reduce_product %12 {axis = 1 : i32} : (tensor<5x8xi1>) -> tensor<5x1xi1>
    %14 = tosa.logical_right_shift %13, %13 : (tensor<5x1xi1>, tensor<5x1xi1>) -> tensor<5x1xi1>
    return %8, %14 : tensor<1x52xf32>, tensor<5x1xi1>
  }
}
