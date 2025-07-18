module {
  func.func @main(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<54x46xi64>) -> (tensor<i64>, tensor<54x46xi64>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %1 = tosa.reverse %arg2 {axis = 1 : i32} : (tensor<54x46xi64>) -> tensor<54x46xi64>
    return %0, %1 : tensor<i64>, tensor<54x46xi64>
  }
}
