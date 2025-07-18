module {
  func.func @main(%arg0: tensor<17x60x32x52x43xf32>, %arg1: tensor<48xi1>, %arg2: tensor<48xi1>) -> (tensor<10x1x11x3x2xf32>, tensor<48xi1>) {
    %0 = tosa.log %arg0 : (tensor<17x60x32x52x43xf32>) -> tensor<17x60x32x52x43xf32>
    %1 = tosa.pow %0, %0 : (tensor<17x60x32x52x43xf32>, tensor<17x60x32x52x43xf32>) -> tensor<17x60x32x52x43xf32>
    %s_2_start = tosa.const_shape {values = dense<[ 7, 4, 8, 16, 14 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %s_2_size = tosa.const_shape {values = dense<[ 10, 1, 11, 3, 2 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %2 = tosa.slice %1, %s_2_start, %s_2_size : (tensor<17x60x32x52x43xf32>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<10x1x11x3x2xf32>
    %3 = tosa.logical_or %arg1, %arg2 : (tensor<48xi1>, tensor<48xi1>) -> tensor<48xi1>
    return %2, %3 : tensor<10x1x11x3x2xf32>, tensor<48xi1>
  }
}
