module {
  func.func @main(%arg0: tensor<88x25xi1>, %arg1: tensor<84x77x43xi32>, %arg2: tensor<84x1x43xi32>, %arg3: tensor<12x93x29x75xf32>) -> (tensor<88x25xi1>, tensor<12x93x29x75xf32>, tensor<7x3x11xi1>) {
    %0 = tosa.logical_not %arg0 : (tensor<88x25xi1>) -> tensor<88x25xi1>
    %1 = tosa.equal %arg1, %arg2 : (tensor<84x77x43xi32>, tensor<84x1x43xi32>) -> tensor<84x77x43xi1>
    %2 = tosa.floor %arg3 : (tensor<12x93x29x75xf32>) -> tensor<12x93x29x75xf32>
    %s_3_start = tosa.const_shape {values = dense<[ 35, 46, 32 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_3_size = tosa.const_shape {values = dense<[ 7, 3, 11 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %3 = tosa.slice %1, %s_3_start, %s_3_size : (tensor<84x77x43xi1>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<7x3x11xi1>
    return %0, %2, %3 : tensor<88x25xi1>, tensor<12x93x29x75xf32>, tensor<7x3x11xi1>
  }
}
