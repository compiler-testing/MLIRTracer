module {
  func.func @main(%arg0: tensor<47xi16>, %arg1: tensor<1x2xi32>, %arg2: tensor<5x3x8x43x73xi8>, %arg3: tensor<5x1x8x43x73xi8>) -> (tensor<47xi16>, tensor<5x3x8x43x73xi1>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<2xindex>} : () -> !tosa.shape<2>
    %pad_const_0 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<47xi16>, !tosa.shape<2>, tensor<1xi16>) -> tensor<47xi16>
    %1 = tosa.greater %arg2, %arg3 : (tensor<5x3x8x43x73xi8>, tensor<5x1x8x43x73xi8>) -> tensor<5x3x8x43x73xi1>
    return %0, %1 : tensor<47xi16>, tensor<5x3x8x43x73xi1>
  }
}
