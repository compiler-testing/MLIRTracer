module {
  func.func @main(%arg0: tensor<68x84x88x55x77x58xf32>, %arg1: tensor<10x23xi1>, %arg2: tensor<1x23xi1>) -> (tensor<2x5x5x3x9x9xi1>, tensor<1x23xi1>, tensor<2x5x5x3x9x9xi1>) {
    %s_0_start = tosa.const_shape {values = dense<[ 53, 55, 5, 52, 64, 14 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_0_size = tosa.const_shape {values = dense<[ 2, 5, 5, 3, 9, 9 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<68x84x88x55x77x58xf32>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<2x5x5x3x9x9xf32>
    %1 = tosa.logical_or %arg1, %arg2 : (tensor<10x23xi1>, tensor<1x23xi1>) -> tensor<10x23xi1>
    %2 = tosa.floor %0 : (tensor<2x5x5x3x9x9xf32>) -> tensor<2x5x5x3x9x9xf32>
    %3 = tosa.greater %2, %0 : (tensor<2x5x5x3x9x9xf32>, tensor<2x5x5x3x9x9xf32>) -> tensor<2x5x5x3x9x9xi1>
    %4 = tosa.clz %3 : (tensor<2x5x5x3x9x9xi1>) -> tensor<2x5x5x3x9x9xi1>
    %5 = tosa.reduce_max %1 {axis = 0 : i32} : (tensor<10x23xi1>) -> tensor<1x23xi1>
    %6 = tosa.equal %0, %0 : (tensor<2x5x5x3x9x9xf32>, tensor<2x5x5x3x9x9xf32>) -> tensor<2x5x5x3x9x9xi1>
    return %4, %5, %6 : tensor<2x5x5x3x9x9xi1>, tensor<1x23xi1>, tensor<2x5x5x3x9x9xi1>
  }
}
