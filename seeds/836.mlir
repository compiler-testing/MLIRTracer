module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<49x2x11x87xi1>) -> (tensor<f32>, tensor<49x2x87xi32>) {
    %0 = tosa.log %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.sigmoid %0 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.tanh %1 : (tensor<f32>) -> tensor<f32>
    %3 = tosa.argmax %arg1 {axis = 2 : i32} : (tensor<49x2x11x87xi1>) -> tensor<49x2x87xi32>
    return %2, %3 : tensor<f32>, tensor<49x2x87xi32>
  }
}
