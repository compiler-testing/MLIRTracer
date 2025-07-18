module {
  func.func @main(%arg0: tensor<14xi1>, %arg1: tensor<1xi1>, %arg2: tensor<f32>) -> (tensor<14xi1>, tensor<f32>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<14xi1>, tensor<1xi1>) -> tensor<14xi1>
    %1 = tosa.rsqrt %arg2 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.tanh %1 : (tensor<f32>) -> tensor<f32>
    return %0, %2 : tensor<14xi1>, tensor<f32>
  }
}
