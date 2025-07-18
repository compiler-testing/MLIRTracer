module {
  func.func @main(%arg0: tensor<76xf32>) -> tensor<228xf32> {
    %0 = tosa.sigmoid %arg0 : (tensor<76xf32>) -> tensor<76xf32>
    %t_1 = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %1 = tosa.tile %0, %t_1 : (tensor<76xf32>, !tosa.shape<1>) -> tensor<228xf32>
    %2 = tosa.sub %1, %1 : (tensor<228xf32>, tensor<228xf32>) -> tensor<228xf32>
    return %2 : tensor<228xf32>
  }
}
