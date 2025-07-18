module {
  func.func @main(%arg0: tensor<15x49x32x70xi16>, %arg1: tensor<27x11x77x70x83x54xi1>, %arg2: tensor<1x11x1x70x1x54xi1>, %arg3: tensor<34x55x38x64x60x96xf32>, %arg4: tensor<1x55x1x1x1x1xf32>) -> (tensor<4x4x9x1xi16>, tensor<27x11x77x70x83x54xi1>, tensor<34x55x38x64x60x96xi1>) {
    %s_0_start = tosa.const_shape {values = dense<[ 2, 11, 4, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_0_size = tosa.const_shape {values = dense<[ 4, 4, 9, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<15x49x32x70xi16>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<4x4x9x1xi16>
    %1 = tosa.logical_and %arg1, %arg2 : (tensor<27x11x77x70x83x54xi1>, tensor<1x11x1x70x1x54xi1>) -> tensor<27x11x77x70x83x54xi1>
    %2 = tosa.logical_left_shift %1, %1 : (tensor<27x11x77x70x83x54xi1>, tensor<27x11x77x70x83x54xi1>) -> tensor<27x11x77x70x83x54xi1>
    %3 = tosa.greater_equal %arg3, %arg4 : (tensor<34x55x38x64x60x96xf32>, tensor<1x55x1x1x1x1xf32>) -> tensor<34x55x38x64x60x96xi1>
    return %0, %2, %3 : tensor<4x4x9x1xi16>, tensor<27x11x77x70x83x54xi1>, tensor<34x55x38x64x60x96xi1>
  }
}
