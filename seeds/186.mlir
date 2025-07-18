module {
  func.func @main(%arg0: tensor<2x74x75x29xi32>, %arg1: tensor<2x1x75x29xi32>, %arg2: tensor<43x79x69x82x36x88xf32>) -> (tensor<40xi32>, tensor<4x9x11x3x8x11xf32>) {
    %0 = tosa.sub %arg0, %arg1 : (tensor<2x74x75x29xi32>, tensor<2x1x75x29xi32>) -> tensor<2x74x75x29xi32>
    %1 = tosa.sub %0, %0 : (tensor<2x74x75x29xi32>, tensor<2x74x75x29xi32>) -> tensor<2x74x75x29xi32>
    %s_2_start = tosa.const_shape {values = dense<[ 1, 2, 1, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_2_size = tosa.const_shape {values = dense<[ 1, 2, 8, 5 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %2 = tosa.slice %1, %s_2_start, %s_2_size : (tensor<2x74x75x29xi32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x2x8x5xi32>
    %3 = tosa.reduce_sum %2 {axis = 1 : i32} : (tensor<1x2x8x5xi32>) -> tensor<1x1x8x5xi32>
    %4 = tosa.add %3, %3 : (tensor<1x1x8x5xi32>, tensor<1x1x8x5xi32>) -> tensor<1x1x8x5xi32>
    %5 = tosa.ceil %arg2 : (tensor<43x79x69x82x36x88xf32>) -> tensor<43x79x69x82x36x88xf32>
    %r_6 = tosa.const_shape {values = dense<[ 40 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %6 = tosa.reshape %4, %r_6 : (tensor<1x1x8x5xi32>, !tosa.shape<1>) -> tensor<40xi32>
    %s_7_start = tosa.const_shape {values = dense<[ 10, 38, 13, 36, 12, 16 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_7_size = tosa.const_shape {values = dense<[ 4, 9, 11, 3, 8, 11 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %7 = tosa.slice %5, %s_7_start, %s_7_size : (tensor<43x79x69x82x36x88xf32>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<4x9x11x3x8x11xf32>
    %8 = tosa.clamp %7 {min_val = -6.000000e+00 : f32, max_val = 9.200000e+01 : f32} : (tensor<4x9x11x3x8x11xf32>) -> tensor<4x9x11x3x8x11xf32>
    return %6, %8 : tensor<40xi32>, tensor<4x9x11x3x8x11xf32>
  }
}
