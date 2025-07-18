module {
  func.func @main(%arg0: tensor<24x89x79x5xf32>, %arg1: tensor<1x89x1x1xf32>, %arg2: tensor<i16>, %arg3: tensor<i16>) -> (tensor<7x12x8x1xf32>, tensor<i16>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<24x89x79x5xf32>, tensor<1x89x1x1xf32>) -> tensor<24x89x79x5xf32>
    %1 = tosa.arithmetic_right_shift %arg2, %arg3 {round = true} : (tensor<i16>, tensor<i16>) -> tensor<i16>
    %in_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.negate %0, %in_zp_2, %out_zp_2 : (tensor<24x89x79x5xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<24x89x79x5xf32>
    %s_3_start = tosa.const_shape {values = dense<[ 17, 14, 1, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_3_size = tosa.const_shape {values = dense<[ 7, 12, 8, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %3 = tosa.slice %2, %s_3_start, %s_3_size : (tensor<24x89x79x5xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<7x12x8x1xf32>
    %4 = tosa.arithmetic_right_shift %1, %1 {round = true} : (tensor<i16>, tensor<i16>) -> tensor<i16>
    %5 = tosa.bitwise_not %4 : (tensor<i16>) -> tensor<i16>
    return %3, %5 : tensor<7x12x8x1xf32>, tensor<i16>
  }
}
