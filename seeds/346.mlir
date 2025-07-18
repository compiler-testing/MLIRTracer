module {
  func.func @main(%arg0: tensor<15xi1>, %arg1: tensor<74x19x59x5x88x57xf32>) -> (tensor<2xi1>, tensor<74x19x59x5x88x57xf32>, tensor<9x4x12x4x5x5xf32>) {
    %s_0_start = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_0_size = tosa.const_shape {values = dense<[ 11 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<15xi1>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<11xi1>
    %s_1_start = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_1_size = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %1 = tosa.slice %0, %s_1_start, %s_1_size : (tensor<11xi1>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<2xi1>
    %2 = tosa.tanh %arg1 : (tensor<74x19x59x5x88x57xf32>) -> tensor<74x19x59x5x88x57xf32>
    %3 = tosa.bitwise_not %1 : (tensor<2xi1>) -> tensor<2xi1>
    %4 = tosa.pow %2, %2 : (tensor<74x19x59x5x88x57xf32>, tensor<74x19x59x5x88x57xf32>) -> tensor<74x19x59x5x88x57xf32>
    %s_5_start = tosa.const_shape {values = dense<[ 32, 15, 36, 1, 2, 52 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_5_size = tosa.const_shape {values = dense<[ 9, 4, 12, 4, 5, 5 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %5 = tosa.slice %2, %s_5_start, %s_5_size : (tensor<74x19x59x5x88x57xf32>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<9x4x12x4x5x5xf32>
    %6 = tosa.sigmoid %4 : (tensor<74x19x59x5x88x57xf32>) -> tensor<74x19x59x5x88x57xf32>
    %7 = tosa.tanh %5 : (tensor<9x4x12x4x5x5xf32>) -> tensor<9x4x12x4x5x5xf32>
    return %3, %6, %7 : tensor<2xi1>, tensor<74x19x59x5x88x57xf32>, tensor<9x4x12x4x5x5xf32>
  }
}
