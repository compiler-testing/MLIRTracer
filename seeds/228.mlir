module {
  func.func @main(%arg0: tensor<27x53xi32>) -> tensor<7x8xi32> {
    %0 = tosa.clz %arg0 : (tensor<27x53xi32>) -> tensor<27x53xi32>
    %1 = tosa.minimum %0, %0 : (tensor<27x53xi32>, tensor<27x53xi32>) -> tensor<27x53xi32>
    %s_2_start = tosa.const_shape {values = dense<[ 20, 25 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_2_size = tosa.const_shape {values = dense<[ 7, 8 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %2 = tosa.slice %1, %s_2_start, %s_2_size : (tensor<27x53xi32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<7x8xi32>
    return %2 : tensor<7x8xi32>
  }
}
