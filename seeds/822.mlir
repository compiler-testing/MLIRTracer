module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<39xi1>, %arg2: tensor<1xi1>) -> (tensor<f32>, tensor<39xi1>) {
    %0 = tosa.exp %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.sigmoid %0 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.logical_or %arg1, %arg2 : (tensor<39xi1>, tensor<1xi1>) -> tensor<39xi1>
    return %1, %2 : tensor<f32>, tensor<39xi1>
  }
}
