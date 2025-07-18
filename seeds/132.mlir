module {
  func.func @main(%arg0: tensor<19x95x49x33x4xf32>, %arg1: tensor<5x2xi64>, %arg2: tensor<92x65xi1>, %arg3: tensor<1x65xi1>) -> (tensor<19x95x49x33x4xf32>, tensor<92x65xi1>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<10xindex>} : () -> !tosa.shape<10>
    %pad_const_0 = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<19x95x49x33x4xf32>, !tosa.shape<10>, tensor<1xf32>) -> tensor<19x95x49x33x4xf32>
    %1 = tosa.logical_or %arg2, %arg3 : (tensor<92x65xi1>, tensor<1x65xi1>) -> tensor<92x65xi1>
    return %0, %1 : tensor<19x95x49x33x4xf32>, tensor<92x65xi1>
  }
}
