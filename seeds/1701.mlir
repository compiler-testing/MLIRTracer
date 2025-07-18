module {
  func.func @main(%arg0: tensor<11xi8>) -> tensor<11xi1> {
    %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<11xi8>) -> tensor<11xi8>
    %1 = tosa.greater %0, %0 : (tensor<11xi8>, tensor<11xi8>) -> tensor<11xi1>
    %t_2 = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %2 = tosa.tile %1, %t_2 : (tensor<11xi1>, !tosa.shape<1>) -> tensor<11xi1>
    return %2 : tensor<11xi1>
  }
}
