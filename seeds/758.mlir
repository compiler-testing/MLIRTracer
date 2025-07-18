module {
  func.func @main(%arg0: tensor<11x97x65x63xi16>, %arg1: tensor<4x2xi32>, %arg2: tensor<51x86xf32>) -> (tensor<11x97x65x63xi16>, tensor<51x86xf32>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<8xindex>} : () -> !tosa.shape<8>
    %pad_const_0 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<11x97x65x63xi16>, !tosa.shape<8>, tensor<1xi16>) -> tensor<11x97x65x63xi16>
    %1 = tosa.reciprocal %arg2 : (tensor<51x86xf32>) -> tensor<51x86xf32>
    %2 = tosa.abs %0 : (tensor<11x97x65x63xi16>) -> tensor<11x97x65x63xi16>
    %3 = tosa.floor %1 : (tensor<51x86xf32>) -> tensor<51x86xf32>
    %4 = tosa.minimum %3, %1 : (tensor<51x86xf32>, tensor<51x86xf32>) -> tensor<51x86xf32>
    return %2, %4 : tensor<11x97x65x63xi16>, tensor<51x86xf32>
  }
}
