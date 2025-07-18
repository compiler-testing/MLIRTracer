module {
  func.func @main(%arg0: tensor<37xf32>, %arg1: tensor<91xi1>, %arg2: tensor<91xi1>) -> (tensor<1xf32>, tensor<37xf32>, tensor<11xi1>, tensor<37xf32>, tensor<i32>) {
    %in_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<37xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<37xf32>
    %1 = tosa.reduce_sum %0 {axis = 0 : i32} : (tensor<37xf32>) -> tensor<1xf32>
    %2 = tosa.sub %1, %1 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %3 = tosa.reduce_sum %2 {axis = 0 : i32} : (tensor<1xf32>) -> tensor<1xf32>
    %4 = tosa.reduce_product %3 {axis = 0 : i32} : (tensor<1xf32>) -> tensor<1xf32>
    %5 = tosa.maximum %4, %3 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %6 = tosa.logical_or %arg1, %arg2 : (tensor<91xi1>, tensor<91xi1>) -> tensor<91xi1>
    %7 = tosa.logical_left_shift %6, %6 : (tensor<91xi1>, tensor<91xi1>) -> tensor<91xi1>
    %8 = tosa.pow %0, %0 : (tensor<37xf32>, tensor<37xf32>) -> tensor<37xf32>
    %9 = tosa.reverse %6 {axis = 0 : i32} : (tensor<91xi1>) -> tensor<91xi1>
    %s_10_start = tosa.const_shape {values = dense<[ 33 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_10_size = tosa.const_shape {values = dense<[ 11 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %10 = tosa.slice %7, %s_10_start, %s_10_size : (tensor<91xi1>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<11xi1>
    %11 = tosa.bitwise_not %9 : (tensor<91xi1>) -> tensor<91xi1>
    %s_12_start = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_12_size = tosa.const_shape {values = dense<[ 5 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %12 = tosa.slice %11, %s_12_start, %s_12_size : (tensor<91xi1>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<5xi1>
    %13 = tosa.pow %0, %0 : (tensor<37xf32>, tensor<37xf32>) -> tensor<37xf32>
    %14 = tosa.argmax %12 {axis = 0 : i32} : (tensor<5xi1>) -> tensor<i32>
    return %5, %8, %10, %13, %14 : tensor<1xf32>, tensor<37xf32>, tensor<11xi1>, tensor<37xf32>, tensor<i32>
  }
}
