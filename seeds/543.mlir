module {
  func.func @main(%arg0: tensor<75x72x36x15xi64>) -> tensor<4x9x1x1xi64> {
    %s_0_start = tosa.const_shape {values = dense<[ 23, 11, 24, 8 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_0_size = tosa.const_shape {values = dense<[ 4, 9, 12, 7 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<75x72x36x15xi64>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<4x9x12x7xi64>
    %1 = tosa.reduce_product %0 {axis = 2 : i32} : (tensor<4x9x12x7xi64>) -> tensor<4x9x1x7xi64>
    %2 = tosa.reduce_min %1 {axis = 3 : i32} : (tensor<4x9x1x7xi64>) -> tensor<4x9x1x1xi64>
    %3 = tosa.minimum %2, %2 : (tensor<4x9x1x1xi64>, tensor<4x9x1x1xi64>) -> tensor<4x9x1x1xi64>
    return %3 : tensor<4x9x1x1xi64>
  }
}
