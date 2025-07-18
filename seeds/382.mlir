module {
  func.func @main(%arg0: tensor<50x21xi1>, %arg1: tensor<1x21xi1>, %arg2: tensor<14x15x86x70xi32>, %arg3: tensor<1x1x86x70xi32>, %arg4: tensor<f32>, %arg5: tensor<f32>) -> (tensor<150x21xi1>, tensor<14x15x86x70xi1>, tensor<f32>) {
    %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<50x21xi1>, tensor<1x21xi1>) -> tensor<50x21xi1>
    %t_1 = tosa.const_shape {values = dense<[ 3, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.tile %0, %t_1 : (tensor<50x21xi1>, !tosa.shape<2>) -> tensor<150x21xi1>
    %2 = tosa.greater_equal %arg2, %arg3 : (tensor<14x15x86x70xi32>, tensor<1x1x86x70xi32>) -> tensor<14x15x86x70xi1>
    %3 = tosa.logical_and %2, %2 : (tensor<14x15x86x70xi1>, tensor<14x15x86x70xi1>) -> tensor<14x15x86x70xi1>
    %4 = tosa.pow %arg4, %arg5 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    return %1, %3, %4 : tensor<150x21xi1>, tensor<14x15x86x70xi1>, tensor<f32>
  }
}
