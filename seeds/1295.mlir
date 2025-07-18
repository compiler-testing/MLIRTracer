module {
  func.func @main(%arg0: tensor<27x76x86xi64>, %arg1: tensor<23x60xf32>) -> (tensor<15x16x1xi1>, tensor<23x60xf32>, tensor<5x8x1xi1>) {
    %s_0_start = tosa.const_shape {values = dense<[ 14, 20, 5 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_0_size = tosa.const_shape {values = dense<[ 5, 8, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<27x76x86xi64>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<5x8x1xi64>
    %1 = tosa.greater %0, %0 : (tensor<5x8x1xi64>, tensor<5x8x1xi64>) -> tensor<5x8x1xi1>
    %2 = tosa.reciprocal %arg1 : (tensor<23x60xf32>) -> tensor<23x60xf32>
    %3 = tosa.bitwise_and %1, %1 : (tensor<5x8x1xi1>, tensor<5x8x1xi1>) -> tensor<5x8x1xi1>
    %t_4 = tosa.const_shape {values = dense<[ 3, 2, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %4 = tosa.tile %3, %t_4 : (tensor<5x8x1xi1>, !tosa.shape<3>) -> tensor<15x16x1xi1>
    %5 = tosa.sigmoid %2 : (tensor<23x60xf32>) -> tensor<23x60xf32>
    %6 = tosa.sub %5, %2 : (tensor<23x60xf32>, tensor<23x60xf32>) -> tensor<23x60xf32>
    %7 = tosa.logical_xor %1, %3 : (tensor<5x8x1xi1>, tensor<5x8x1xi1>) -> tensor<5x8x1xi1>
    return %4, %6, %7 : tensor<15x16x1xi1>, tensor<23x60xf32>, tensor<5x8x1xi1>
  }
}
