module {
  func.func @main(%arg0: tensor<86xi32>, %arg1: tensor<37xi32>, %arg2: tensor<f32>) -> (tensor<f32>, tensor<123xi32>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<86xi32>, tensor<37xi32>) -> tensor<123xi32>
    %1 = tosa.exp %arg2 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.intdiv %0, %0 : (tensor<123xi32>, tensor<123xi32>) -> tensor<123xi32>
    return %1, %2 : tensor<f32>, tensor<123xi32>
  }
}
