module {
  func.func @main(%arg0: tensor<f32>) -> tensor<i1> {
    %0 = tosa.floor %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.floor %0 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.greater_equal %1, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    return %2 : tensor<i1>
  }
}
