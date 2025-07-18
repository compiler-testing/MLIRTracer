module {
  func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = tosa.exp %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.floor %0 : (tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
}
