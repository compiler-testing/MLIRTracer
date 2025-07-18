module {
  func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = tosa.identity %arg0 : (tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}
