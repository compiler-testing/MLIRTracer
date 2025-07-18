module {
  func.func @main(%arg0: tensor<40x20x41xi1>, %arg1: tensor<84x25x29x89x61xf32>, %arg2: tensor<84x25x1x89x1xf32>) -> (tensor<40x1x41xi1>, tensor<8x11x3x12x6xf32>) {
    %0 = tosa.reduce_product %arg0 {axis = 1 : i32} : (tensor<40x20x41xi1>) -> tensor<40x1x41xi1>
    %1 = tosa.logical_right_shift %0, %0 : (tensor<40x1x41xi1>, tensor<40x1x41xi1>) -> tensor<40x1x41xi1>
    %2 = tosa.pow %arg1, %arg2 : (tensor<84x25x29x89x61xf32>, tensor<84x25x1x89x1xf32>) -> tensor<84x25x29x89x61xf32>
    %s_3_start = tosa.const_shape {values = dense<[ 76, 14, 26, 12, 48 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %s_3_size = tosa.const_shape {values = dense<[ 8, 11, 3, 12, 6 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %3 = tosa.slice %2, %s_3_start, %s_3_size : (tensor<84x25x29x89x61xf32>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<8x11x3x12x6xf32>
    return %1, %3 : tensor<40x1x41xi1>, tensor<8x11x3x12x6xf32>
  }
}
