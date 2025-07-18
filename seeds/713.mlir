module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<82x77x81xi1>) -> (tensor<f32>, tensor<1x77x81xi1>) {
    %0 = tosa.floor %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.reduce_any %arg1 {axis = 0 : i32} : (tensor<82x77x81xi1>) -> tensor<1x77x81xi1>
    %2 = tosa.logical_right_shift %1, %1 : (tensor<1x77x81xi1>, tensor<1x77x81xi1>) -> tensor<1x77x81xi1>
    %3 = tosa.bitwise_or %1, %2 : (tensor<1x77x81xi1>, tensor<1x77x81xi1>) -> tensor<1x77x81xi1>
    return %0, %3 : tensor<f32>, tensor<1x77x81xi1>
  }
}
