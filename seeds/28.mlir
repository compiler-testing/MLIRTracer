module {
  func.func @main(%arg0: tensor<50xi1>, %arg1: tensor<f32>) -> (tensor<i32>, tensor<f32>) {
    %0 = tosa.reduce_any %arg0 {axis = 0 : i32} : (tensor<50xi1>) -> tensor<1xi1>
    %1 = tosa.argmax %0 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<i32>
    %2 = tosa.ceil %arg1 : (tensor<f32>) -> tensor<f32>
    %3 = tosa.sub %2, %2 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    return %1, %3 : tensor<i32>, tensor<f32>
  }
}
