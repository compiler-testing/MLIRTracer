module {
  func.func @main(%arg0: tensor<73x32xi32>, %arg1: tensor<f32>) -> (tensor<73x1xi32>, tensor<f32>) {
    %0 = tosa.reduce_sum %arg0 {axis = 1 : i32} : (tensor<73x32xi32>) -> tensor<73x1xi32>
    %1 = tosa.ceil %arg1 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.tanh %1 : (tensor<f32>) -> tensor<f32>
    return %0, %2 : tensor<73x1xi32>, tensor<f32>
  }
}
