module {
  func.func @main(%arg0: tensor<25x62x32x47x11xi64>) -> tensor<1000x2xi1> {
    %s_0_start = tosa.const_shape {values = dense<[ 4, 8, 18, 24, 6 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %s_0_size = tosa.const_shape {values = dense<[ 5, 10, 4, 2, 5 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<25x62x32x47x11xi64>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<5x10x4x2x5xi64>
    %1 = tosa.greater %0, %0 : (tensor<5x10x4x2x5xi64>, tensor<5x10x4x2x5xi64>) -> tensor<5x10x4x2x5xi1>
    %2 = tosa.logical_and %1, %1 : (tensor<5x10x4x2x5xi1>, tensor<5x10x4x2x5xi1>) -> tensor<5x10x4x2x5xi1>
    %r_3 = tosa.const_shape {values = dense<[ 1000, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %3 = tosa.reshape %2, %r_3 : (tensor<5x10x4x2x5xi1>, !tosa.shape<2>) -> tensor<1000x2xi1>
    %4 = tosa.logical_xor %3, %3 : (tensor<1000x2xi1>, tensor<1000x2xi1>) -> tensor<1000x2xi1>
    return %4 : tensor<1000x2xi1>
  }
}
