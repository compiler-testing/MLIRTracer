module {
  func.func @main(%arg0: tensor<38x28xi8>, %arg1: tensor<38x1xi8>, %arg2: tensor<11xi1>, %arg3: tensor<11xi1>) -> (tensor<11xi1>, tensor<1x1064x1xi8>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<38x28xi8>, tensor<38x1xi8>) -> tensor<38x28xi8>
    %1 = tosa.logical_and %arg2, %arg3 : (tensor<11xi1>, tensor<11xi1>) -> tensor<11xi1>
    %r_2 = tosa.const_shape {values = dense<[ 1, 1064, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %2 = tosa.reshape %0, %r_2 : (tensor<38x28xi8>, !tosa.shape<3>) -> tensor<1x1064x1xi8>
    return %1, %2 : tensor<11xi1>, tensor<1x1064x1xi8>
  }
}
