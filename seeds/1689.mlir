module {
  func.func @main(%arg0: tensor<15x33x53x42x66xi1>, %arg1: tensor<21x11xf32>) -> (tensor<4x9x9x6x7xi1>, tensor<21x11xf32>) {
    %0 = tosa.logical_not %arg0 : (tensor<15x33x53x42x66xi1>) -> tensor<15x33x53x42x66xi1>
    %1 = tosa.logical_not %0 : (tensor<15x33x53x42x66xi1>) -> tensor<15x33x53x42x66xi1>
    %s_2_start = tosa.const_shape {values = dense<[ 11, 9, 13, 11, 6 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %s_2_size = tosa.const_shape {values = dense<[ 4, 9, 9, 6, 7 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %2 = tosa.slice %1, %s_2_start, %s_2_size : (tensor<15x33x53x42x66xi1>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<4x9x9x6x7xi1>
    %3 = tosa.rsqrt %arg1 : (tensor<21x11xf32>) -> tensor<21x11xf32>
    return %2, %3 : tensor<4x9x9x6x7xi1>, tensor<21x11xf32>
  }
}
