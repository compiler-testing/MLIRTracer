module {
  func.func @main(%arg0: tensor<88xi32>) -> tensor<1x1x1xi32> {
    %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<88xi32>) -> tensor<1xi32>
    %r_1 = tosa.const_shape {values = dense<[ 1, 1, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.reshape %0, %r_1 : (tensor<1xi32>, !tosa.shape<3>) -> tensor<1x1x1xi32>
    return %1 : tensor<1x1x1xi32>
  }
}
