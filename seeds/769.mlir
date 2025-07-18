module {
  func.func @main(%arg0: tensor<7x92x70x20x34xf32>, %arg1: tensor<60x32x55xi1>, %arg2: tensor<1x32x55xi1>) -> (tensor<7x92x70x20x34xf32>, tensor<7x92x70x20x34xf32>, tensor<60x32x55xi1>, tensor<7x92x70x20x34xf32>, tensor<4x3x12xi1>) {
    %0 = tosa.rsqrt %arg0 : (tensor<7x92x70x20x34xf32>) -> tensor<7x92x70x20x34xf32>
    %1 = tosa.minimum %0, %0 : (tensor<7x92x70x20x34xf32>, tensor<7x92x70x20x34xf32>) -> tensor<7x92x70x20x34xf32>
    %2 = tosa.logical_right_shift %arg1, %arg2 : (tensor<60x32x55xi1>, tensor<1x32x55xi1>) -> tensor<60x32x55xi1>
    %3 = tosa.logical_or %2, %2 : (tensor<60x32x55xi1>, tensor<60x32x55xi1>) -> tensor<60x32x55xi1>
    %4 = tosa.reduce_min %3 {axis = 1 : i32} : (tensor<60x32x55xi1>) -> tensor<60x1x55xi1>
    %5 = tosa.floor %1 : (tensor<7x92x70x20x34xf32>) -> tensor<7x92x70x20x34xf32>
    %6 = tosa.bitwise_not %2 : (tensor<60x32x55xi1>) -> tensor<60x32x55xi1>
    %7 = tosa.sigmoid %0 : (tensor<7x92x70x20x34xf32>) -> tensor<7x92x70x20x34xf32>
    %8 = tosa.rsqrt %7 : (tensor<7x92x70x20x34xf32>) -> tensor<7x92x70x20x34xf32>
    %9 = tosa.arithmetic_right_shift %4, %4 {round = false} : (tensor<60x1x55xi1>, tensor<60x1x55xi1>) -> tensor<60x1x55xi1>
    %s_10_start = tosa.const_shape {values = dense<[ 39, 0, 43 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_10_size = tosa.const_shape {values = dense<[ 4, 3, 12 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %10 = tosa.slice %9, %s_10_start, %s_10_size : (tensor<60x1x55xi1>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<4x3x12xi1>
    %11 = tosa.logical_left_shift %6, %3 : (tensor<60x32x55xi1>, tensor<60x32x55xi1>) -> tensor<60x32x55xi1>
    %12 = tosa.sub %10, %10 : (tensor<4x3x12xi1>, tensor<4x3x12xi1>) -> tensor<4x3x12xi1>
    %13 = tosa.sigmoid %0 : (tensor<7x92x70x20x34xf32>) -> tensor<7x92x70x20x34xf32>
    %14 = tosa.logical_not %12 : (tensor<4x3x12xi1>) -> tensor<4x3x12xi1>
    %15 = tosa.logical_right_shift %14, %14 : (tensor<4x3x12xi1>, tensor<4x3x12xi1>) -> tensor<4x3x12xi1>
    return %5, %8, %11, %13, %15 : tensor<7x92x70x20x34xf32>, tensor<7x92x70x20x34xf32>, tensor<60x32x55xi1>, tensor<7x92x70x20x34xf32>, tensor<4x3x12xi1>
  }
}
