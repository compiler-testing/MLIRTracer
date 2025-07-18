module {
  func.func @main(%arg0: tensor<75x73x92xi32>, %arg1: tensor<75x1x92xi32>, %arg2: tensor<f32>) -> (tensor<f32>, tensor<75x1x92xi32>) {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<75x73x92xi32>, tensor<75x1x92xi32>) -> tensor<75x73x92xi32>
    %1 = tosa.exp %arg2 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.reduce_max %0 {axis = 1 : i32} : (tensor<75x73x92xi32>) -> tensor<75x1x92xi32>
    %3 = tosa.intdiv %2, %2 : (tensor<75x1x92xi32>, tensor<75x1x92xi32>) -> tensor<75x1x92xi32>
    return %1, %3 : tensor<f32>, tensor<75x1x92xi32>
  }
}
