module {
  func.func @main(%arg0: tensor<16xf32>, %arg1: tensor<1xf32>, %arg2: tensor<14x51x14xi32>, %arg3: tensor<14x51x1xi32>) -> (tensor<1x1x1x1xf32>, tensor<14x51x14xi32>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<16xf32>, tensor<1xf32>) -> tensor<16xf32>
    %1 = tosa.ceil %0 : (tensor<16xf32>) -> tensor<16xf32>
    %2 = tosa.concat %1, %0 {axis = 0 : i32} : (tensor<16xf32>, tensor<16xf32>) -> tensor<32xf32>
    %3 = tosa.pow %2, %2 : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
    %4 = tosa.ceil %3 : (tensor<32xf32>) -> tensor<32xf32>
    %5 = tosa.intdiv %arg2, %arg3 : (tensor<14x51x14xi32>, tensor<14x51x1xi32>) -> tensor<14x51x14xi32>
    %in_zp_6 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_6 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %6 = tosa.negate %4, %in_zp_6, %out_zp_6 : (tensor<32xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<32xf32>
    %s_7_start = tosa.const_shape {values = dense<[ 13 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_7_size = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %7 = tosa.slice %6, %s_7_start, %s_7_size : (tensor<32xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<1xf32>
    %8 = tosa.maximum %5, %5 : (tensor<14x51x14xi32>, tensor<14x51x14xi32>) -> tensor<14x51x14xi32>
    %9 = tosa.ceil %7 : (tensor<1xf32>) -> tensor<1xf32>
    %r_10 = tosa.const_shape {values = dense<[ 1, 1, 1, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %10 = tosa.reshape %9, %r_10 : (tensor<1xf32>, !tosa.shape<4>) -> tensor<1x1x1x1xf32>
    %in_zp_11 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_11 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %11 = tosa.negate %8, %in_zp_11, %out_zp_11 : (tensor<14x51x14xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<14x51x14xi32>
    %12 = tosa.bitwise_not %11 : (tensor<14x51x14xi32>) -> tensor<14x51x14xi32>
    return %10, %12 : tensor<1x1x1x1xf32>, tensor<14x51x14xi32>
  }
}
