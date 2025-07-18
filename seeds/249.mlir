module {
  func.func @main(%arg0: tensor<57x34xi1>, %arg1: tensor<1x1xi1>, %arg2: tensor<77x90x46xi8>, %arg3: tensor<77x1x1xi8>, %arg4: tensor<37x26x60x38xi8>, %arg5: tensor<37x26x60x1xi8>) -> (tensor<57x34xi1>, tensor<318780xi1>, tensor<37x26x60x38xi1>) {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<57x34xi1>, tensor<1x1xi1>) -> tensor<57x34xi1>
    %1 = tosa.greater_equal %arg2, %arg3 : (tensor<77x90x46xi8>, tensor<77x1x1xi8>) -> tensor<77x90x46xi1>
    %r_2 = tosa.const_shape {values = dense<[ 318780 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %2 = tosa.reshape %1, %r_2 : (tensor<77x90x46xi1>, !tosa.shape<1>) -> tensor<318780xi1>
    %3 = tosa.greater_equal %arg4, %arg5 : (tensor<37x26x60x38xi8>, tensor<37x26x60x1xi8>) -> tensor<37x26x60x38xi1>
    return %0, %2, %3 : tensor<57x34xi1>, tensor<318780xi1>, tensor<37x26x60x38xi1>
  }
}
