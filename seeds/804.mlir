module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<3xi32>, %arg2: tensor<17x25x22xi1>) -> (tensor<f32>, tensor<f32>, tensor<1x25x22xi1>, tensor<4xi32>, tensor<1xi1>) {
    %0 = tosa.ceil %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.reduce_min %arg1 {axis = 0 : i32} : (tensor<3xi32>) -> tensor<1xi32>
    %2 = tosa.pow %0, %0 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %3 = tosa.ceil %2 : (tensor<f32>) -> tensor<f32>
    %4 = tosa.ceil %3 : (tensor<f32>) -> tensor<f32>
    %s_5_start = tosa.const_shape {values = dense<[ 0 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_5_size = tosa.const_shape {values = dense<[ 7 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %5 = tosa.slice %1, %s_5_start, %s_5_size : (tensor<1xi32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<7xi32>
    %6 = tosa.logical_right_shift %5, %5 : (tensor<7xi32>, tensor<7xi32>) -> tensor<7xi32>
    %7 = tosa.arithmetic_right_shift %6, %5 {round = true} : (tensor<7xi32>, tensor<7xi32>) -> tensor<7xi32>
    %8 = tosa.reduce_all %arg2 {axis = 0 : i32} : (tensor<17x25x22xi1>) -> tensor<1x25x22xi1>
    %9 = tosa.bitwise_or %7, %7 : (tensor<7xi32>, tensor<7xi32>) -> tensor<7xi32>
    %10 = tosa.logical_left_shift %9, %7 : (tensor<7xi32>, tensor<7xi32>) -> tensor<7xi32>
    %11 = tosa.maximum %10, %6 : (tensor<7xi32>, tensor<7xi32>) -> tensor<7xi32>
    %12 = tosa.clamp %11 {min_val = 51 : i32, max_val = 52 : i32} : (tensor<7xi32>) -> tensor<7xi32>
    %13 = tosa.rsqrt %3 : (tensor<f32>) -> tensor<f32>
    %14 = tosa.bitwise_not %1 : (tensor<1xi32>) -> tensor<1xi32>
    %15 = tosa.logical_not %8 : (tensor<1x25x22xi1>) -> tensor<1x25x22xi1>
    %s_16_start = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_16_size = tosa.const_shape {values = dense<[ 4 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %16 = tosa.slice %12, %s_16_start, %s_16_size : (tensor<7xi32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<4xi32>
    %17 = tosa.greater %14, %1 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    return %4, %13, %15, %16, %17 : tensor<f32>, tensor<f32>, tensor<1x25x22xi1>, tensor<4xi32>, tensor<1xi1>
  }
}
