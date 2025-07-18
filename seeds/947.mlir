module {
  func.func @main(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<91xf32>) -> (tensor<i64>, tensor<i32>, tensor<1xf32>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %1 = tosa.reciprocal %arg2 : (tensor<91xf32>) -> tensor<91xf32>
    %2 = tosa.reduce_min %1 {axis = 0 : i32} : (tensor<91xf32>) -> tensor<1xf32>
    %3 = tosa.reduce_sum %1 {axis = 0 : i32} : (tensor<91xf32>) -> tensor<1xf32>
    %4 = tosa.argmax %3 {axis = 0 : i32} : (tensor<1xf32>) -> tensor<i32>
    %5 = tosa.abs %4 : (tensor<i32>) -> tensor<i32>
    %6 = tosa.pow %2, %2 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    return %0, %5, %6 : tensor<i64>, tensor<i32>, tensor<1xf32>
  }
}
