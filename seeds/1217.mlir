module {
  func.func @main(%arg0: tensor<32x53x95x27xi16>, %arg1: tensor<4x2xi32>, %arg2: tensor<93x37xf32>) -> (tensor<32x53x95x27xi16>, tensor<186x74xf32>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<8xindex>} : () -> !tosa.shape<8>
    %pad_const_0 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<32x53x95x27xi16>, !tosa.shape<8>, tensor<1xi16>) -> tensor<32x53x95x27xi16>
    %1 = tosa.reciprocal %arg2 : (tensor<93x37xf32>) -> tensor<93x37xf32>
    %t_2 = tosa.const_shape {values = dense<[ 2, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %2 = tosa.tile %1, %t_2 : (tensor<93x37xf32>, !tosa.shape<2>) -> tensor<186x74xf32>
    return %0, %2 : tensor<32x53x95x27xi16>, tensor<186x74xf32>
  }
}
