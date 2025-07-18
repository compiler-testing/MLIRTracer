module {
  func.func @main(%arg0: tensor<55x43x44xi32>) -> tensor<7x1x11xi32> {
    %s_0_start = tosa.const_shape {values = dense<[ 10, 15, 11 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_0_size = tosa.const_shape {values = dense<[ 7, 12, 11 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<55x43x44xi32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<7x12x11xi32>
    %1 = tosa.clamp %0 {min_val = -54 : i32, max_val = -48 : i32} : (tensor<7x12x11xi32>) -> tensor<7x12x11xi32>
    %2 = tosa.minimum %1, %0 : (tensor<7x12x11xi32>, tensor<7x12x11xi32>) -> tensor<7x12x11xi32>
    %3 = tosa.reduce_max %2 {axis = 1 : i32} : (tensor<7x12x11xi32>) -> tensor<7x1x11xi32>
    %4 = tosa.logical_right_shift %3, %3 : (tensor<7x1x11xi32>, tensor<7x1x11xi32>) -> tensor<7x1x11xi32>
    %5 = tosa.clz %4 : (tensor<7x1x11xi32>) -> tensor<7x1x11xi32>
    return %5 : tensor<7x1x11xi32>
  }
}
