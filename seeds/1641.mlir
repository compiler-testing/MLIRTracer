module {
  func.func @main(%arg0: tensor<47x44x14x4xf32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> (tensor<1x1x14x4xf32>, tensor<i64>) {
    %0 = tosa.reduce_sum %arg0 {axis = 1 : i32} : (tensor<47x44x14x4xf32>) -> tensor<47x1x14x4xf32>
    %1 = tosa.reduce_sum %0 {axis = 0 : i32} : (tensor<47x1x14x4xf32>) -> tensor<1x1x14x4xf32>
    %2 = tosa.bitwise_or %arg1, %arg2 : (tensor<i64>, tensor<i64>) -> tensor<i64>
    return %1, %2 : tensor<1x1x14x4xf32>, tensor<i64>
  }
}
