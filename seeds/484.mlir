module {
  func.func @main(%arg0: tensor<95x9x40x7xi1>) -> tensor<4x3x10x5xi1> {
    %s_0_start = tosa.const_shape {values = dense<[ 66, 6, 30, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_0_size = tosa.const_shape {values = dense<[ 4, 3, 10, 5 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<95x9x40x7xi1>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<4x3x10x5xi1>
    return %0 : tensor<4x3x10x5xi1>
  }
}
