module {
  func.func @main(%arg0: tensor<4x41x14x9xf32>, %arg1: tensor<1x41x14x1xf32>) -> tensor<3x6888x1xf32> {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<4x41x14x9xf32>, tensor<1x41x14x1xf32>) -> tensor<4x41x14x9xf32>
    %r_1 = tosa.const_shape {values = dense<[ 3, 6888, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.reshape %0, %r_1 : (tensor<4x41x14x9xf32>, !tosa.shape<3>) -> tensor<3x6888x1xf32>
    return %1 : tensor<3x6888x1xf32>
  }
}
