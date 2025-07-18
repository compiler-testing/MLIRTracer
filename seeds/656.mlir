module {
  func.func @main(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<83x33x48xf32>, %arg3: tensor<83x33x1xf32>) -> (tensor<i32>, tensor<9x3x11xi1>, tensor<5x5x11xi32>, tensor<83x33x48xf32>) {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %1 = tosa.pow %arg2, %arg3 : (tensor<83x33x48xf32>, tensor<83x33x1xf32>) -> tensor<83x33x48xf32>
    %s_2_start = tosa.const_shape {values = dense<[ 69, 30, 14 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_2_size = tosa.const_shape {values = dense<[ 9, 3, 11 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %2 = tosa.slice %1, %s_2_start, %s_2_size : (tensor<83x33x48xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<9x3x11xf32>
    %3 = tosa.greater %2, %2 : (tensor<9x3x11xf32>, tensor<9x3x11xf32>) -> tensor<9x3x11xi1>
    %4 = tosa.rsqrt %1 : (tensor<83x33x48xf32>) -> tensor<83x33x48xf32>
    %5 = tosa.reduce_sum %1 {axis = 2 : i32} : (tensor<83x33x48xf32>) -> tensor<83x33x1xf32>
    %6 = tosa.bitwise_xor %0, %0 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %r_7 = tosa.const_shape {values = dense<[ 1, 3, 913, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %7 = tosa.reshape %5, %r_7 : (tensor<83x33x1xf32>, !tosa.shape<4>) -> tensor<1x3x913x1xf32>
    %8 = tosa.reciprocal %7 : (tensor<1x3x913x1xf32>) -> tensor<1x3x913x1xf32>
    %9 = tosa.clz %3 : (tensor<9x3x11xi1>) -> tensor<9x3x11xi1>
    %s_10_start = tosa.const_shape {values = dense<[ 0, 0, 1, 0 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_10_size = tosa.const_shape {values = dense<[ 5, 5, 11, 6 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %10 = tosa.slice %8, %s_10_start, %s_10_size : (tensor<1x3x913x1xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<5x5x11x6xf32>
    %11 = tosa.maximum %10, %10 : (tensor<5x5x11x6xf32>, tensor<5x5x11x6xf32>) -> tensor<5x5x11x6xf32>
    %12 = tosa.argmax %11 {axis = 3 : i32} : (tensor<5x5x11x6xf32>) -> tensor<5x5x11xi32>
    %13 = tosa.reciprocal %4 : (tensor<83x33x48xf32>) -> tensor<83x33x48xf32>
    %14 = tosa.sub %13, %4 : (tensor<83x33x48xf32>, tensor<83x33x48xf32>) -> tensor<83x33x48xf32>
    return %6, %9, %12, %14 : tensor<i32>, tensor<9x3x11xi1>, tensor<5x5x11xi32>, tensor<83x33x48xf32>
  }
}
