module {
  func.func @main(%arg0: tensor<98x62xi1>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<98x62xi1>) {
    %0 = tosa.bitwise_not %arg0 : (tensor<98x62xi1>) -> tensor<98x62xi1>
    %1 = tosa.reciprocal %arg1 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.reverse %0 {axis = 1 : i32} : (tensor<98x62xi1>) -> tensor<98x62xi1>
    %3 = tosa.reciprocal %1 : (tensor<f32>) -> tensor<f32>
    %4 = tosa.logical_not %2 : (tensor<98x62xi1>) -> tensor<98x62xi1>
    return %3, %4 : tensor<f32>, tensor<98x62xi1>
  }
}
