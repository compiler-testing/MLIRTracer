module {
  func.func @main(%arg0: tensor<50x53x74x42x97x60xi32>, %arg1: tensor<1xi1>, %arg2: tensor<1xi1>, %arg3: tensor<41x47xf32>) -> (tensor<1x4x5x11x11x8xi32>, tensor<41x47xf32>, tensor<1xi1>) {
    %0 = tosa.identity %arg0 : (tensor<50x53x74x42x97x60xi32>) -> tensor<50x53x74x42x97x60xi32>
    %1 = tosa.logical_xor %arg1, %arg2 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %2 = tosa.floor %arg3 : (tensor<41x47xf32>) -> tensor<41x47xf32>
    %s_3_start = tosa.const_shape {values = dense<[ 11, 17, 18, 7, 35, 18 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_3_size = tosa.const_shape {values = dense<[ 1, 4, 5, 11, 11, 8 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %3 = tosa.slice %0, %s_3_start, %s_3_size : (tensor<50x53x74x42x97x60xi32>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<1x4x5x11x11x8xi32>
    %4 = tosa.logical_right_shift %3, %3 : (tensor<1x4x5x11x11x8xi32>, tensor<1x4x5x11x11x8xi32>) -> tensor<1x4x5x11x11x8xi32>
    %5 = tosa.pow %2, %2 : (tensor<41x47xf32>, tensor<41x47xf32>) -> tensor<41x47xf32>
    %6 = tosa.logical_and %1, %1 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %7 = tosa.logical_xor %1, %6 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    return %4, %5, %7 : tensor<1x4x5x11x11x8xi32>, tensor<41x47xf32>, tensor<1xi1>
  }
}
