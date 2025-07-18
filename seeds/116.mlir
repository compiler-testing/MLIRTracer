module {
  func.func @main(%arg0: tensor<95x21xi32>, %arg1: tensor<1x21xi32>) -> tensor<35x19x1x3xi32> {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<95x21xi32>, tensor<1x21xi32>) -> tensor<95x21xi32>
    %r_1 = tosa.const_shape {values = dense<[ 35, 19, 1, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %1 = tosa.reshape %0, %r_1 : (tensor<95x21xi32>, !tosa.shape<4>) -> tensor<35x19x1x3xi32>
    return %1 : tensor<35x19x1x3xi32>
  }
}
