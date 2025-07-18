module {
  func.func @main(%arg0: tensor<57x51x79x12xf32>, %arg1: tensor<57x51x79x1xf32>) -> tensor<11x10x8x8xf32> {
    %0 = tosa.sub %arg0, %arg1 : (tensor<57x51x79x12xf32>, tensor<57x51x79x1xf32>) -> tensor<57x51x79x12xf32>
    %s_1_start = tosa.const_shape {values = dense<[ 46, 41, 19, 4 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_1_size = tosa.const_shape {values = dense<[ 11, 10, 8, 8 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %1 = tosa.slice %0, %s_1_start, %s_1_size : (tensor<57x51x79x12xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<11x10x8x8xf32>
    %2 = tosa.ceil %1 : (tensor<11x10x8x8xf32>) -> tensor<11x10x8x8xf32>
    return %2 : tensor<11x10x8x8xf32>
  }
}
