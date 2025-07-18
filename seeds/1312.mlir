module {
  func.func @main(%arg0: tensor<72xi1>, %arg1: tensor<f32>) -> (tensor<1xi1>, tensor<f32>) {
    %0 = tosa.reduce_all %arg0 {axis = 0 : i32} : (tensor<72xi1>) -> tensor<1xi1>
    %1 = tosa.ceil %arg1 : (tensor<f32>) -> tensor<f32>
    return %0, %1 : tensor<1xi1>, tensor<f32>
  }
}
