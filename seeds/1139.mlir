module {
  func.func @main(%arg0: tensor<96xi16>, %arg1: tensor<f32>, %arg2: tensor<19x43x70xi1>, %arg3: tensor<1x43x70xi1>) -> (tensor<i32>, tensor<19x43x70xi1>, tensor<f32>) {
    %0 = tosa.argmax %arg0 {axis = 0 : i32} : (tensor<96xi16>) -> tensor<i32>
    %1 = tosa.floor %arg1 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.logical_xor %arg2, %arg3 : (tensor<19x43x70xi1>, tensor<1x43x70xi1>) -> tensor<19x43x70xi1>
    %3 = tosa.reciprocal %1 : (tensor<f32>) -> tensor<f32>
    return %0, %2, %3 : tensor<i32>, tensor<19x43x70xi1>, tensor<f32>
  }
}
