module {
  func.func @main(%arg0: tensor<21x16xi32>, %arg1: tensor<f32>) -> (tensor<21xi32>, tensor<f32>) {
    %0 = tosa.argmax %arg0 {axis = 1 : i32} : (tensor<21x16xi32>) -> tensor<21xi32>
    %1 = tosa.sigmoid %arg1 : (tensor<f32>) -> tensor<f32>
    return %0, %1 : tensor<21xi32>, tensor<f32>
  }
}
