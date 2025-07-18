module {
  func.func @main(%arg0: tensor<3x51x32x71xi8>, %arg1: tensor<1x1x32x1xi8>) -> tensor<9x4x8x6xi1> {
    %0 = tosa.greater %arg0, %arg1 : (tensor<3x51x32x71xi8>, tensor<1x1x32x1xi8>) -> tensor<3x51x32x71xi1>
    %1 = tosa.reduce_product %0 {axis = 0 : i32} : (tensor<3x51x32x71xi1>) -> tensor<1x51x32x71xi1>
    %2 = tosa.reduce_product %1 {axis = 3 : i32} : (tensor<1x51x32x71xi1>) -> tensor<1x51x32x1xi1>
    %s_3_start = tosa.const_shape {values = dense<[ 0, 1, 1, 0 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_3_size = tosa.const_shape {values = dense<[ 9, 4, 8, 6 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %3 = tosa.slice %2, %s_3_start, %s_3_size : (tensor<1x51x32x1xi1>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<9x4x8x6xi1>
    %4 = tosa.logical_and %3, %3 : (tensor<9x4x8x6xi1>, tensor<9x4x8x6xi1>) -> tensor<9x4x8x6xi1>
    return %4 : tensor<9x4x8x6xi1>
  }
}
