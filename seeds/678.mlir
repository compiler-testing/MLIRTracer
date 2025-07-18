module {
  func.func @main(%arg0: tensor<34x73x41x16x90xf32>, %arg1: tensor<5x2xi64>) -> tensor<34x73x41x16x90xf32> {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<10xindex>} : () -> !tosa.shape<10>
    %pad_const_0 = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<34x73x41x16x90xf32>, !tosa.shape<10>, tensor<1xf32>) -> tensor<34x73x41x16x90xf32>
    %1 = tosa.minimum %0, %0 : (tensor<34x73x41x16x90xf32>, tensor<34x73x41x16x90xf32>) -> tensor<34x73x41x16x90xf32>
    return %1 : tensor<34x73x41x16x90xf32>
  }
}
