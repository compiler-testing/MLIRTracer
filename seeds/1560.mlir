module {
  func.func @main(%arg0: tensor<70x50x63xi16>) -> tensor<70x150x126xi16> {
    %t_0 = tosa.const_shape {values = dense<[ 1, 3, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = tosa.tile %arg0, %t_0 : (tensor<70x50x63xi16>, !tosa.shape<3>) -> tensor<70x150x126xi16>
    return %0 : tensor<70x150x126xi16>
  }
}
