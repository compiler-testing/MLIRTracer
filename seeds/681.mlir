module {
  func.func @main(%arg0: tensor<22x33xi16>) -> tensor<363x2xi16> {
    %r_0 = tosa.const_shape {values = dense<[ 363, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<22x33xi16>, !tosa.shape<2>) -> tensor<363x2xi16>
    return %0 : tensor<363x2xi16>
  }
}
