module {
  func.func @main(%arg0: tensor<84x10x76xf32>, %arg1: tensor<84x10x76xf32>, %arg2: tensor<i64>, %arg3: tensor<i64>, %arg4: tensor<i32>, %arg5: tensor<i32>, %arg6: tensor<32x90x32xi1>) -> (tensor<84x10x76xf32>, tensor<i64>, tensor<i32>, tensor<32x90x32xi1>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<84x10x76xf32>, tensor<84x10x76xf32>) -> tensor<84x10x76xf32>
    %1 = tosa.logical_right_shift %arg2, %arg3 : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %2 = tosa.bitwise_or %1, %1 : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %3 = tosa.bitwise_xor %1, %2 : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %4 = tosa.identity %3 : (tensor<i64>) -> tensor<i64>
    %5 = tosa.tanh %0 : (tensor<84x10x76xf32>) -> tensor<84x10x76xf32>
    %6 = tosa.sigmoid %5 : (tensor<84x10x76xf32>) -> tensor<84x10x76xf32>
    %7 = tosa.sub %4, %2 : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %8 = tosa.intdiv %arg4, %arg5 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %9 = tosa.logical_not %arg6 : (tensor<32x90x32xi1>) -> tensor<32x90x32xi1>
    return %6, %7, %8, %9 : tensor<84x10x76xf32>, tensor<i64>, tensor<i32>, tensor<32x90x32xi1>
  }
}
