module {
  func.func @main(%arg0: tensor<83xi32>, %arg1: tensor<83xi32>) -> tensor<1x2x4xi1> {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<83xi32>, tensor<83xi32>) -> tensor<83xi1>
    %1 = tosa.reverse %0 {axis = 0 : i32} : (tensor<83xi1>) -> tensor<83xi1>
    %s_2_start = tosa.const_shape {values = dense<[ 50 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_2_size = tosa.const_shape {values = dense<[ 8 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %2 = tosa.slice %1, %s_2_start, %s_2_size : (tensor<83xi1>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<8xi1>
    %3 = tosa.logical_left_shift %2, %2 : (tensor<8xi1>, tensor<8xi1>) -> tensor<8xi1>
    %r_4 = tosa.const_shape {values = dense<[ 1, 2, 4 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %4 = tosa.reshape %3, %r_4 : (tensor<8xi1>, !tosa.shape<3>) -> tensor<1x2x4xi1>
    return %4 : tensor<1x2x4xi1>
  }
}
