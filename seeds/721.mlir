module {
  func.func @main(%arg0: tensor<i1>) -> tensor<1x1xi32> {
    %r_0 = tosa.const_shape {values = dense<[ 1, 1, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<i1>, !tosa.shape<3>) -> tensor<1x1x1xi1>
    %1 = tosa.argmax %0 {axis = 0 : i32} : (tensor<1x1x1xi1>) -> tensor<1x1xi32>
    return %1 : tensor<1x1xi32>
  }
}
