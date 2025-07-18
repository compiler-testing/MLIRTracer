module {
  func.func @main(%arg0: tensor<2x20x25xi1>) -> tensor<2x7x1xi1> {
    %s_0_start = tosa.const_shape {values = dense<[ 0, 1, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_0_size = tosa.const_shape {values = dense<[ 2, 7, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<2x20x25xi1>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<2x7x3xi1>
    %1 = tosa.logical_not %0 : (tensor<2x7x3xi1>) -> tensor<2x7x3xi1>
    %2 = tosa.identity %1 : (tensor<2x7x3xi1>) -> tensor<2x7x3xi1>
    %3 = tosa.logical_right_shift %2, %0 : (tensor<2x7x3xi1>, tensor<2x7x3xi1>) -> tensor<2x7x3xi1>
    %4 = tosa.reduce_all %3 {axis = 2 : i32} : (tensor<2x7x3xi1>) -> tensor<2x7x1xi1>
    return %4 : tensor<2x7x1xi1>
  }
}
