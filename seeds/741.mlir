module {
  func.func @main(%arg0: tensor<89x34x7x67x25xi1>, %arg1: tensor<1x34x7x1x1xi1>, %arg2: tensor<f32>, %arg3: tensor<4x60x65x3xf32>, %arg4: tensor<4x1x1x3xf32>) -> (tensor<105910x335xi1>, tensor<i1>, tensor<f32>, tensor<4x60x65x3xf32>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<89x34x7x67x25xi1>, tensor<1x34x7x1x1xi1>) -> tensor<89x34x7x67x25xi1>
    %1 = tosa.logical_left_shift %0, %0 : (tensor<89x34x7x67x25xi1>, tensor<89x34x7x67x25xi1>) -> tensor<89x34x7x67x25xi1>
    %r_2 = tosa.const_shape {values = dense<[ 105910, 335 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %2 = tosa.reshape %1, %r_2 : (tensor<89x34x7x67x25xi1>, !tosa.shape<2>) -> tensor<105910x335xi1>
    %3 = tosa.ceil %arg2 : (tensor<f32>) -> tensor<f32>
    %4 = tosa.greater %3, %3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %5 = tosa.ceil %3 : (tensor<f32>) -> tensor<f32>
    %6 = tosa.minimum %arg3, %arg4 : (tensor<4x60x65x3xf32>, tensor<4x1x1x3xf32>) -> tensor<4x60x65x3xf32>
    return %2, %4, %5, %6 : tensor<105910x335xi1>, tensor<i1>, tensor<f32>, tensor<4x60x65x3xf32>
  }
}
