module {
  func.func @main(%arg0: tensor<55x67x84x96x5x26xi8>, %arg1: tensor<6x2xi32>, %arg2: tensor<93x9x73x42xi1>, %arg3: tensor<93x9x73x42xi1>) -> (tensor<55x67x84x96x5x26xi8>, tensor<93x9x73x42xi1>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<12xindex>} : () -> !tosa.shape<12>
    %pad_const_0 = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<55x67x84x96x5x26xi8>, !tosa.shape<12>, tensor<1xi8>) -> tensor<55x67x84x96x5x26xi8>
    %1 = tosa.logical_xor %arg2, %arg3 : (tensor<93x9x73x42xi1>, tensor<93x9x73x42xi1>) -> tensor<93x9x73x42xi1>
    return %0, %1 : tensor<55x67x84x96x5x26xi8>, tensor<93x9x73x42xi1>
  }
}
