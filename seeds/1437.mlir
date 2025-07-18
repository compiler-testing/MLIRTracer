module {
  func.func @main(%arg0: tensor<62xi1>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<1xi1>) {
    %0 = tosa.reduce_any %arg0 {axis = 0 : i32} : (tensor<62xi1>) -> tensor<1xi1>
    %1 = tosa.ceil %arg1 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.logical_left_shift %0, %0 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    return %1, %2 : tensor<f32>, tensor<1xi1>
  }
}
