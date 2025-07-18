module {
  func.func @main(%arg0: tensor<i32>, %arg1: tensor<29x21x49x86xi64>) -> (tensor<i32>, tensor<29x21x1x86xi64>) {
    %0 = tosa.clz %arg0 : (tensor<i32>) -> tensor<i32>
    %1 = tosa.bitwise_xor %0, %0 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = tosa.abs %1 : (tensor<i32>) -> tensor<i32>
    %3 = tosa.reduce_min %arg1 {axis = 2 : i32} : (tensor<29x21x49x86xi64>) -> tensor<29x21x1x86xi64>
    return %2, %3 : tensor<i32>, tensor<29x21x1x86xi64>
  }
}
