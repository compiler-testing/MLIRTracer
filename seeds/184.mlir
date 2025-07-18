module {
  func.func @main(%arg0: tensor<37x11x59xi8>, %arg1: tensor<37x1x59xi8>) -> (tensor<2x59x1xi1>, tensor<111x11x118xi8>, tensor<111x11x118xi1>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<37x11x59xi8>, tensor<37x1x59xi8>) -> tensor<37x11x59xi8>
    %t_1 = tosa.const_shape {values = dense<[ 3, 1, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.tile %0, %t_1 : (tensor<37x11x59xi8>, !tosa.shape<3>) -> tensor<111x11x118xi8>
    %2 = tosa.greater_equal %1, %1 : (tensor<111x11x118xi8>, tensor<111x11x118xi8>) -> tensor<111x11x118xi1>
    %3 = tosa.reduce_all %2 {axis = 0 : i32} : (tensor<111x11x118xi1>) -> tensor<1x11x118xi1>
    %4 = tosa.reduce_max %3 {axis = 1 : i32} : (tensor<1x11x118xi1>) -> tensor<1x1x118xi1>
    %r_5 = tosa.const_shape {values = dense<[ 2, 59, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %5 = tosa.reshape %4, %r_5 : (tensor<1x1x118xi1>, !tosa.shape<3>) -> tensor<2x59x1xi1>
    %6 = tosa.minimum %1, %1 : (tensor<111x11x118xi8>, tensor<111x11x118xi8>) -> tensor<111x11x118xi8>
    %7 = tosa.greater_equal %1, %1 : (tensor<111x11x118xi8>, tensor<111x11x118xi8>) -> tensor<111x11x118xi1>
    return %5, %6, %7 : tensor<2x59x1xi1>, tensor<111x11x118xi8>, tensor<111x11x118xi1>
  }
}
