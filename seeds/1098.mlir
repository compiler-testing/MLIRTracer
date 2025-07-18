module {
  func.func @main(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<43xf32>, %arg3: tensor<1xf32>, %arg4: tensor<82x15x66xi32>, %arg5: tensor<82x1x66xi32>, %arg6: tensor<f32>) -> (tensor<43xi1>, tensor<82x15x66xi32>, tensor<1x1xi1>, tensor<f32>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1 = tosa.equal %arg2, %arg3 : (tensor<43xf32>, tensor<1xf32>) -> tensor<43xi1>
    %2 = tosa.logical_and %0, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %3 = tosa.maximum %arg4, %arg5 : (tensor<82x15x66xi32>, tensor<82x1x66xi32>) -> tensor<82x15x66xi32>
    %4 = tosa.ceil %arg6 : (tensor<f32>) -> tensor<f32>
    %r_5 = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %5 = tosa.reshape %2, %r_5 : (tensor<i1>, !tosa.shape<2>) -> tensor<1x1xi1>
    %6 = tosa.reciprocal %4 : (tensor<f32>) -> tensor<f32>
    return %1, %3, %5, %6 : tensor<43xi1>, tensor<82x15x66xi32>, tensor<1x1xi1>, tensor<f32>
  }
}
