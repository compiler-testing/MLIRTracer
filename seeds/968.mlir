module {
  func.func @main(%arg0: tensor<13x10xi8>) -> tensor<1x26x1x5xi8> {
    %r_0 = tosa.const_shape {values = dense<[ 1, 26, 1, 5 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<13x10xi8>, !tosa.shape<4>) -> tensor<1x26x1x5xi8>
    return %0 : tensor<1x26x1x5xi8>
  }
}
