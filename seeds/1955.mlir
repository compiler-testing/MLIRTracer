module {
  func.func @main(%arg0: tensor<2x23x67xi1>, %arg1: tensor<64x43x21x75xi32>, %arg2: tensor<1x43x1x75xi32>, %arg3: tensor<11x9x56xi64>, %arg4: tensor<11x1x1xi64>) -> (tensor<64x43x21x75xi1>, tensor<8x11x6xi1>, tensor<11x9x56xi1>) {
    %s_0_start = tosa.const_shape {values = dense<[ 0, 1, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_0_size = tosa.const_shape {values = dense<[ 5, 7, 6 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<2x23x67xi1>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<5x7x6xi1>
    %1 = tosa.greater %arg1, %arg2 : (tensor<64x43x21x75xi32>, tensor<1x43x1x75xi32>) -> tensor<64x43x21x75xi1>
    %s_2_start = tosa.const_shape {values = dense<[ 0, 0, 0 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_2_size = tosa.const_shape {values = dense<[ 8, 11, 6 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %2 = tosa.slice %0, %s_2_start, %s_2_size : (tensor<5x7x6xi1>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<8x11x6xi1>
    %3 = tosa.equal %arg3, %arg4 : (tensor<11x9x56xi64>, tensor<11x1x1xi64>) -> tensor<11x9x56xi1>
    %4 = tosa.abs %3 : (tensor<11x9x56xi1>) -> tensor<11x9x56xi1>
    return %1, %2, %4 : tensor<64x43x21x75xi1>, tensor<8x11x6xi1>, tensor<11x9x56xi1>
  }
}
