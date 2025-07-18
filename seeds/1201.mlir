module {
  func.func @main(%arg0: tensor<56xi8>, %arg1: tensor<56xi8>) -> tensor<56xi8> {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = true} : (tensor<56xi8>, tensor<56xi8>) -> tensor<56xi8>
    %1 = tosa.maximum %0, %0 : (tensor<56xi8>, tensor<56xi8>) -> tensor<56xi8>
    %t_2 = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %2 = tosa.tile %1, %t_2 : (tensor<56xi8>, !tosa.shape<1>) -> tensor<56xi8>
    %3 = tosa.maximum %2, %0 : (tensor<56xi8>, tensor<56xi8>) -> tensor<56xi8>
    %4 = tosa.bitwise_or %3, %2 : (tensor<56xi8>, tensor<56xi8>) -> tensor<56xi8>
    %5 = tosa.minimum %4, %1 : (tensor<56xi8>, tensor<56xi8>) -> tensor<56xi8>
    return %5 : tensor<56xi8>
  }
}
