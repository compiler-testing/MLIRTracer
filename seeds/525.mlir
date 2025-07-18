module {
  func.func @main(%arg0: tensor<57x89xf32>, %arg1: tensor<57x1xf32>) -> tensor<12x11xf32> {
    %0 = tosa.pow %arg0, %arg1 : (tensor<57x89xf32>, tensor<57x1xf32>) -> tensor<57x89xf32>
    %s_1_start = tosa.const_shape {values = dense<[ 38, 37 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_1_size = tosa.const_shape {values = dense<[ 12, 11 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.slice %0, %s_1_start, %s_1_size : (tensor<57x89xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<12x11xf32>
    return %1 : tensor<12x11xf32>
  }
}
