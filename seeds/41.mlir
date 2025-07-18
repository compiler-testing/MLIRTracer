module {
  func.func @main(%arg0: tensor<59x17x83x53xi8>, %arg1: tensor<4x2xi32>) -> tensor<59x17x83x53xi8> {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<8xindex>} : () -> !tosa.shape<8>
    %pad_const_0 = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<59x17x83x53xi8>, !tosa.shape<8>, tensor<1xi8>) -> tensor<59x17x83x53xi8>
    return %0 : tensor<59x17x83x53xi8>
  }
}
