module {
  func.func @main(%arg0: tensor<50x44x15x46x77x93xi64>, %arg1: tensor<1x1x1x46x77x1xi64>, %arg2: tensor<12x62x60x34xf32>, %arg3: tensor<72x40x60x7xi1>, %arg4: tensor<75x8x69x49x2x39xi32>, %arg5: tensor<75x1x69x1x2x39xi32>) -> (tensor<12x62x60x34xf32>, tensor<72x40x60x1xi1>, tensor<75x8x69x49x2x39xi32>, tensor<8x1x2x3x12x8xi64>, tensor<12x62x60x34xf32>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<50x44x15x46x77x93xi64>, tensor<1x1x1x46x77x1xi64>) -> tensor<50x44x15x46x77x93xi64>
    %s_1_start = tosa.const_shape {values = dense<[ 42, 31, 13, 6, 31, 26 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_1_size = tosa.const_shape {values = dense<[ 8, 1, 2, 3, 12, 8 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %1 = tosa.slice %0, %s_1_start, %s_1_size : (tensor<50x44x15x46x77x93xi64>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<8x1x2x3x12x8xi64>
    %2 = tosa.ceil %arg2 : (tensor<12x62x60x34xf32>) -> tensor<12x62x60x34xf32>
    %3 = tosa.reduce_any %arg3 {axis = 3 : i32} : (tensor<72x40x60x7xi1>) -> tensor<72x40x60x1xi1>
    %4 = tosa.logical_left_shift %1, %1 : (tensor<8x1x2x3x12x8xi64>, tensor<8x1x2x3x12x8xi64>) -> tensor<8x1x2x3x12x8xi64>
    %5 = tosa.clamp %2 {min_val = -5.000000e+01 : f32, max_val = -1.600000e+01 : f32} : (tensor<12x62x60x34xf32>) -> tensor<12x62x60x34xf32>
    %6 = tosa.logical_not %3 : (tensor<72x40x60x1xi1>) -> tensor<72x40x60x1xi1>
    %7 = tosa.intdiv %arg4, %arg5 : (tensor<75x8x69x49x2x39xi32>, tensor<75x1x69x1x2x39xi32>) -> tensor<75x8x69x49x2x39xi32>
    %8 = tosa.logical_right_shift %4, %1 : (tensor<8x1x2x3x12x8xi64>, tensor<8x1x2x3x12x8xi64>) -> tensor<8x1x2x3x12x8xi64>
    %9 = tosa.rsqrt %2 : (tensor<12x62x60x34xf32>) -> tensor<12x62x60x34xf32>
    return %5, %6, %7, %8, %9 : tensor<12x62x60x34xf32>, tensor<72x40x60x1xi1>, tensor<75x8x69x49x2x39xi32>, tensor<8x1x2x3x12x8xi64>, tensor<12x62x60x34xf32>
  }
}
