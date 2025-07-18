module {
  func.func @main(%arg0: tensor<11x66x96xi32>, %arg1: tensor<1x66x96xi32>) -> tensor<1x6x2xi32> {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<11x66x96xi32>, tensor<1x66x96xi32>) -> tensor<11x66x96xi32>
    %1 = tosa.add %0, %0 : (tensor<11x66x96xi32>, tensor<11x66x96xi32>) -> tensor<11x66x96xi32>
    %s_2_start = tosa.const_shape {values = dense<[ 2, 11, 8 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_2_size = tosa.const_shape {values = dense<[ 1, 6, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %2 = tosa.slice %1, %s_2_start, %s_2_size : (tensor<11x66x96xi32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<1x6x2xi32>
    return %2 : tensor<1x6x2xi32>
  }
}
