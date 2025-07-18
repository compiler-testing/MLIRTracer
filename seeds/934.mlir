module {
  func.func @main(%arg0: tensor<77x73x34xi32>, %arg1: tensor<1x1x34xi32>, %arg2: tensor<58x35x87x96xf32>) -> (tensor<2618x73xi32>, tensor<1x1x2x5xf32>) {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<77x73x34xi32>, tensor<1x1x34xi32>) -> tensor<77x73x34xi32>
    %1 = tosa.sigmoid %arg2 : (tensor<58x35x87x96xf32>) -> tensor<58x35x87x96xf32>
    %in_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.negate %1, %in_zp_2, %out_zp_2 : (tensor<58x35x87x96xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<58x35x87x96xf32>
    %3 = tosa.bitwise_and %0, %0 : (tensor<77x73x34xi32>, tensor<77x73x34xi32>) -> tensor<77x73x34xi32>
    %s_4_start = tosa.const_shape {values = dense<[ 20, 25, 35, 26 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_4_size = tosa.const_shape {values = dense<[ 1, 6, 2, 5 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %4 = tosa.slice %2, %s_4_start, %s_4_size : (tensor<58x35x87x96xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x6x2x5xf32>
    %r_5 = tosa.const_shape {values = dense<[ 2618, 73 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %5 = tosa.reshape %3, %r_5 : (tensor<77x73x34xi32>, !tosa.shape<2>) -> tensor<2618x73xi32>
    %6 = tosa.reduce_max %4 {axis = 1 : i32} : (tensor<1x6x2x5xf32>) -> tensor<1x1x2x5xf32>
    %7 = tosa.exp %6 : (tensor<1x1x2x5xf32>) -> tensor<1x1x2x5xf32>
    return %5, %7 : tensor<2618x73xi32>, tensor<1x1x2x5xf32>
  }
}
