module {
  func.func @main(%arg0: tensor<86x33x5x49xf32>) -> tensor<11x12x6x10xf32> {
    %0 = tosa.log %arg0 : (tensor<86x33x5x49xf32>) -> tensor<86x33x5x49xf32>
    %s_1_start = tosa.const_shape {values = dense<[ 40, 21, 0, 39 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_1_size = tosa.const_shape {values = dense<[ 11, 12, 6, 10 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %1 = tosa.slice %0, %s_1_start, %s_1_size : (tensor<86x33x5x49xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<11x12x6x10xf32>
    return %1 : tensor<11x12x6x10xf32>
  }
}
