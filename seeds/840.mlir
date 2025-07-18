module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<81x69x46xi1>) -> (tensor<f32>, tensor<81x2x46xi1>) {
    %0 = tosa.ceil %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.reduce_any %arg1 {axis = 1 : i32} : (tensor<81x69x46xi1>) -> tensor<81x1x46xi1>
    %2 = tosa.ceil %0 : (tensor<f32>) -> tensor<f32>
    %3 = tosa.tanh %2 : (tensor<f32>) -> tensor<f32>
    %4 = tosa.concat %1, %1 {axis = 1 : i32} : (tensor<81x1x46xi1>, tensor<81x1x46xi1>) -> tensor<81x2x46xi1>
    return %3, %4 : tensor<f32>, tensor<81x2x46xi1>
  }
}
