module {
  func.func @main(%arg0: tensor<94x47x46x44xi1>, %arg1: tensor<94x1x46x44xi1>) -> tensor<8942032xi1> {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = true} : (tensor<94x47x46x44xi1>, tensor<94x1x46x44xi1>) -> tensor<94x47x46x44xi1>
    %r_1 = tosa.const_shape {values = dense<[ 8942032 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %1 = tosa.reshape %0, %r_1 : (tensor<94x47x46x44xi1>, !tosa.shape<1>) -> tensor<8942032xi1>
    return %1 : tensor<8942032xi1>
  }
}
