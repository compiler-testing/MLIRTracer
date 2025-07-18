module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = tosa.pow %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %1 = tosa.log %0 : (tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
}
