module {
  func.func @main(%arg0: tensor<19x58x93xi64>, %arg1: tensor<19x72x93xi64>, %arg2: tensor<f32>) -> (tensor<19x130x93xi64>, tensor<f32>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 1 : i32} : (tensor<19x58x93xi64>, tensor<19x72x93xi64>) -> tensor<19x130x93xi64>
    %1 = tosa.ceil %arg2 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.pow %1, %1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    return %0, %2 : tensor<19x130x93xi64>, tensor<f32>
  }
}
