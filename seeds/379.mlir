module {
  func.func @main(%arg0: tensor<92x67x94x94x18xi64>) -> tensor<1738248x564xi64> {
    %r_0 = tosa.const_shape {values = dense<[ 1738248, 564 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<92x67x94x94x18xi64>, !tosa.shape<2>) -> tensor<1738248x564xi64>
    %1 = tosa.logical_left_shift %0, %0 : (tensor<1738248x564xi64>, tensor<1738248x564xi64>) -> tensor<1738248x564xi64>
    return %1 : tensor<1738248x564xi64>
  }
}
