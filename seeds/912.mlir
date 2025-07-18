module {
  func.func @main(%arg0: tensor<92x11x81x46x43x49xi16>, %arg1: tensor<6x2xi64>, %arg2: tensor<35xi32>, %arg3: tensor<35xi32>) -> (tensor<92x11x81x46x43x49xi16>, tensor<35xi32>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<12xindex>} : () -> !tosa.shape<12>
    %pad_const_0 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<92x11x81x46x43x49xi16>, !tosa.shape<12>, tensor<1xi16>) -> tensor<92x11x81x46x43x49xi16>
    %1 = tosa.maximum %arg2, %arg3 : (tensor<35xi32>, tensor<35xi32>) -> tensor<35xi32>
    return %0, %1 : tensor<92x11x81x46x43x49xi16>, tensor<35xi32>
  }
}
