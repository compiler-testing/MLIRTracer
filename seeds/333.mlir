module {
  func.func @main(%arg0: tensor<51x2x19x72xi32>) -> tensor<12x3876x3xi32> {
    %0 = tosa.abs %arg0 : (tensor<51x2x19x72xi32>) -> tensor<51x2x19x72xi32>
    %r_1 = tosa.const_shape {values = dense<[ 12, 3876, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.reshape %0, %r_1 : (tensor<51x2x19x72xi32>, !tosa.shape<3>) -> tensor<12x3876x3xi32>
    return %1 : tensor<12x3876x3xi32>
  }
}
