module {
  func.func @main(%arg0: tensor<f32>) -> (tensor<f32>, tensor<i1>) {
    %0 = tosa.floor %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.greater %0, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2 = tosa.ceil %0 : (tensor<f32>) -> tensor<f32>
    %3 = tosa.logical_right_shift %1, %1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %4 = tosa.abs %3 : (tensor<i1>) -> tensor<i1>
    %5 = tosa.logical_or %4, %3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    return %2, %5 : tensor<f32>, tensor<i1>
  }
}
