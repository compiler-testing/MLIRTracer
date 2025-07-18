module {
  func.func @main(%arg0: tensor<38xi64>, %arg1: tensor<3xi64>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<i32>, tensor<41xi64>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<38xi64>, tensor<3xi64>) -> tensor<41xi64>
    %1 = tosa.abs %0 : (tensor<41xi64>) -> tensor<41xi64>
    %2 = tosa.bitwise_or %1, %0 : (tensor<41xi64>, tensor<41xi64>) -> tensor<41xi64>
    %3 = tosa.intdiv %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %4 = tosa.bitwise_xor %2, %0 : (tensor<41xi64>, tensor<41xi64>) -> tensor<41xi64>
    return %3, %4 : tensor<i32>, tensor<41xi64>
  }
}
