module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<36x7x9x98xf32>, %arg3: tensor<36x7x3x98xf32>) -> (tensor<36x7x12x98xf32>, tensor<f32>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %1 = tosa.concat %arg2, %arg3 {axis = 2 : i32} : (tensor<36x7x9x98xf32>, tensor<36x7x3x98xf32>) -> tensor<36x7x12x98xf32>
    %2 = tosa.pow %0, %0 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    return %1, %2 : tensor<36x7x12x98xf32>, tensor<f32>
  }
}
