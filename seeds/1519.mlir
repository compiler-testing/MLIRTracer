module {
  func.func @main(%arg0: tensor<62x90x50x81x89xi1>, %arg1: tensor<52xi8>, %arg2: tensor<11x29x75xf32>) -> (tensor<62x90x50x81x89xi1>, tensor<104xi8>, tensor<11x29x75xf32>) {
    %0 = tosa.logical_not %arg0 : (tensor<62x90x50x81x89xi1>) -> tensor<62x90x50x81x89xi1>
    %t_1 = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %1 = tosa.tile %arg1, %t_1 : (tensor<52xi8>, !tosa.shape<1>) -> tensor<104xi8>
    %2 = tosa.log %arg2 : (tensor<11x29x75xf32>) -> tensor<11x29x75xf32>
    return %0, %1, %2 : tensor<62x90x50x81x89xi1>, tensor<104xi8>, tensor<11x29x75xf32>
  }
}
