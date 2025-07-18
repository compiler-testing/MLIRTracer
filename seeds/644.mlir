module {
  func.func @main(%arg0: tensor<84xi1>) -> tensor<3x2x14xi1> {
    %0 = tosa.logical_not %arg0 : (tensor<84xi1>) -> tensor<84xi1>
    %r_1 = tosa.const_shape {values = dense<[ 3, 2, 14 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.reshape %0, %r_1 : (tensor<84xi1>, !tosa.shape<3>) -> tensor<3x2x14xi1>
    return %1 : tensor<3x2x14xi1>
  }
}
