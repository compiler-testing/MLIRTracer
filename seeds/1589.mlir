module {
  func.func @main(%arg0: tensor<56xi16>, %arg1: tensor<1x2xi32>) -> tensor<28x1x2xi16> {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<2xindex>} : () -> !tosa.shape<2>
    %pad_const_0 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<56xi16>, !tosa.shape<2>, tensor<1xi16>) -> tensor<56xi16>
    %1 = tosa.clz %0 : (tensor<56xi16>) -> tensor<56xi16>
    %r_2 = tosa.const_shape {values = dense<[ 28, 1, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %2 = tosa.reshape %1, %r_2 : (tensor<56xi16>, !tosa.shape<3>) -> tensor<28x1x2xi16>
    return %2 : tensor<28x1x2xi16>
  }
}
