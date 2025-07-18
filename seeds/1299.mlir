module {
  func.func @main(%arg0: tensor<42x71x85x80x90xi32>, %arg1: tensor<42x71x85x1x90xi32>, %arg2: tensor<7x54x14xi1>, %arg3: tensor<1x1x1xi1>, %arg4: tensor<98x11x91x98x30x83xf32>) -> (tensor<2x12x1x10x10xi32>, tensor<7x54x14xi1>, tensor<98x11x91x98x30x83xf32>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<42x71x85x80x90xi32>, tensor<42x71x85x1x90xi32>) -> tensor<42x71x85x80x90xi32>
    %s_1_start = tosa.const_shape {values = dense<[ 15, 9, 21, 21, 2 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %s_1_size = tosa.const_shape {values = dense<[ 2, 12, 1, 10, 10 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %1 = tosa.slice %0, %s_1_start, %s_1_size : (tensor<42x71x85x80x90xi32>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<2x12x1x10x10xi32>
    %2 = tosa.logical_and %arg2, %arg3 : (tensor<7x54x14xi1>, tensor<1x1x1xi1>) -> tensor<7x54x14xi1>
    %3 = tosa.floor %arg4 : (tensor<98x11x91x98x30x83xf32>) -> tensor<98x11x91x98x30x83xf32>
    return %1, %2, %3 : tensor<2x12x1x10x10xi32>, tensor<7x54x14xi1>, tensor<98x11x91x98x30x83xf32>
  }
}
