module {
  func.func @main(%arg0: tensor<77x11x53x30x87x3xi16>, %arg1: tensor<6x2xi64>) -> tensor<77x11x53x30x87x3xi16> {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<12xindex>} : () -> !tosa.shape<12>
    %pad_const_0 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<77x11x53x30x87x3xi16>, !tosa.shape<12>, tensor<1xi16>) -> tensor<77x11x53x30x87x3xi16>
    return %0 : tensor<77x11x53x30x87x3xi16>
  }
}
