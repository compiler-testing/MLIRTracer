module {
  func.func @main(%arg0: tensor<32xf32>, %arg1: tensor<4x26x37x8x5x62xi1>, %arg2: tensor<4x1x1x1x1x1xi1>) -> (tensor<12x10x2x8x7x9xi1>, tensor<1xf32>) {
    %0 = tosa.log %arg0 : (tensor<32xf32>) -> tensor<32xf32>
    %in_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.negate %0, %in_zp_1, %out_zp_1 : (tensor<32xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<32xf32>
    %2 = tosa.maximum %1, %0 : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
    %3 = tosa.logical_and %arg1, %arg2 : (tensor<4x26x37x8x5x62xi1>, tensor<4x1x1x1x1x1xi1>) -> tensor<4x26x37x8x5x62xi1>
    %s_4_start = tosa.const_shape {values = dense<[ 0, 2, 2, 0, 0, 2 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_4_size = tosa.const_shape {values = dense<[ 12, 10, 2, 8, 7, 9 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %4 = tosa.slice %3, %s_4_start, %s_4_size : (tensor<4x26x37x8x5x62xi1>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<12x10x2x8x7x9xi1>
    %5 = tosa.clamp %2 {min_val = 3.900000e+01 : f32, max_val = 6.800000e+01 : f32} : (tensor<32xf32>) -> tensor<32xf32>
    %6 = tosa.abs %4 : (tensor<12x10x2x8x7x9xi1>) -> tensor<12x10x2x8x7x9xi1>
    %7 = tosa.reverse %5 {axis = 0 : i32} : (tensor<32xf32>) -> tensor<32xf32>
    %8 = tosa.reduce_sum %7 {axis = 0 : i32} : (tensor<32xf32>) -> tensor<1xf32>
    %9 = tosa.log %8 : (tensor<1xf32>) -> tensor<1xf32>
    %10 = tosa.ceil %9 : (tensor<1xf32>) -> tensor<1xf32>
    return %6, %10 : tensor<12x10x2x8x7x9xi1>, tensor<1xf32>
  }
}
