module {
  func.func @main(%arg0: tensor<96x30x30x51x74x96xi16>, %arg1: tensor<1x30x30x1x1x96xi16>, %arg2: tensor<51xf32>, %arg3: tensor<i1>, %arg4: tensor<i1>) -> (tensor<96x30x30x51x74x96xi16>, tensor<i1>, tensor<1x1x1xf32>, tensor<1xf32>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<96x30x30x51x74x96xi16>, tensor<1x30x30x1x1x96xi16>) -> tensor<96x30x30x51x74x96xi16>
    %1 = tosa.rsqrt %arg2 : (tensor<51xf32>) -> tensor<51xf32>
    %2 = tosa.reduce_product %1 {axis = 0 : i32} : (tensor<51xf32>) -> tensor<1xf32>
    %3 = tosa.reduce_min %1 {axis = 0 : i32} : (tensor<51xf32>) -> tensor<1xf32>
    %in_zp_4 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_4 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4 = tosa.negate %2, %in_zp_4, %out_zp_4 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %5 = tosa.rsqrt %4 : (tensor<1xf32>) -> tensor<1xf32>
    %6 = tosa.sub %5, %2 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %7 = tosa.logical_right_shift %0, %0 : (tensor<96x30x30x51x74x96xi16>, tensor<96x30x30x51x74x96xi16>) -> tensor<96x30x30x51x74x96xi16>
    %8 = tosa.logical_or %arg3, %arg4 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %r_9 = tosa.const_shape {values = dense<[ 1, 1, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %9 = tosa.reshape %3, %r_9 : (tensor<1xf32>, !tosa.shape<3>) -> tensor<1x1x1xf32>
    %10 = tosa.pow %6, %4 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    return %7, %8, %9, %10 : tensor<96x30x30x51x74x96xi16>, tensor<i1>, tensor<1x1x1xf32>, tensor<1xf32>
  }
}
