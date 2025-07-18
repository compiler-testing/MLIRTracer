module {
  func.func @main(%arg0: tensor<38x80x1x36x50x97xi16>, %arg1: tensor<23x75x46x83x7xf32>) -> (tensor<8x11x7x5x6x10xi16>, tensor<23x75x46x83x7xf32>, tensor<23x75x46x83x7xf32>) {
    %0 = tosa.abs %arg0 : (tensor<38x80x1x36x50x97xi16>) -> tensor<38x80x1x36x50x97xi16>
    %1 = tosa.rsqrt %arg1 : (tensor<23x75x46x83x7xf32>) -> tensor<23x75x46x83x7xf32>
    %s_2_start = tosa.const_shape {values = dense<[ 30, 35, 0, 26, 5, 28 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_2_size = tosa.const_shape {values = dense<[ 8, 11, 7, 5, 6, 10 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %2 = tosa.slice %0, %s_2_start, %s_2_size : (tensor<38x80x1x36x50x97xi16>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<8x11x7x5x6x10xi16>
    %3 = tosa.bitwise_or %2, %2 : (tensor<8x11x7x5x6x10xi16>, tensor<8x11x7x5x6x10xi16>) -> tensor<8x11x7x5x6x10xi16>
    %4 = tosa.ceil %1 : (tensor<23x75x46x83x7xf32>) -> tensor<23x75x46x83x7xf32>
    %5 = tosa.abs %1 : (tensor<23x75x46x83x7xf32>) -> tensor<23x75x46x83x7xf32>
    return %3, %4, %5 : tensor<8x11x7x5x6x10xi16>, tensor<23x75x46x83x7xf32>, tensor<23x75x46x83x7xf32>
  }
}
