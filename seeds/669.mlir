module {
  func.func @main(%arg0: tensor<91x89x10xi16>, %arg1: tensor<3x2xi32>, %arg2: tensor<69xf32>) -> (tensor<91x89x10xi16>, tensor<69xf32>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<6xindex>} : () -> !tosa.shape<6>
    %pad_const_0 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<91x89x10xi16>, !tosa.shape<6>, tensor<1xi16>) -> tensor<91x89x10xi16>
    %1 = tosa.log %arg2 : (tensor<69xf32>) -> tensor<69xf32>
    return %0, %1 : tensor<91x89x10xi16>, tensor<69xf32>
  }
}
