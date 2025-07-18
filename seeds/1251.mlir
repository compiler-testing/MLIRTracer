module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<i1>, %arg2: tensor<i1>) -> (tensor<i1>, tensor<f32>) {
    %0 = tosa.log %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.logical_and %arg1, %arg2 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %2 = tosa.identity %1 : (tensor<i1>) -> tensor<i1>
    %3 = tosa.logical_and %2, %1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %4 = tosa.pow %0, %0 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    return %3, %4 : tensor<i1>, tensor<f32>
  }
}
