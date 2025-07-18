module {
  func.func @main(%arg0: tensor<28x4xf32>) -> tensor<2x3xf32> {
    %s_0_start = tosa.const_shape {values = dense<[ 2, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_0_size = tosa.const_shape {values = dense<[ 2, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<28x4xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
