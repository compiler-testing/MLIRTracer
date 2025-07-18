module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<64x32x62x68xi32>) -> (tensor<f32>, tensor<64x32x136xi32>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %1 = tosa.argmax %arg2 {axis = 2 : i32} : (tensor<64x32x62x68xi32>) -> tensor<64x32x68xi32>
    %2 = tosa.concat %1, %1 {axis = 2 : i32} : (tensor<64x32x68xi32>, tensor<64x32x68xi32>) -> tensor<64x32x136xi32>
    return %0, %2 : tensor<f32>, tensor<64x32x136xi32>
  }
}
