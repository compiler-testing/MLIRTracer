module {
  func.func @main(%arg0: tensor<4xi1>) -> tensor<i32> {
    %0 = tosa.argmax %arg0 {axis = 0 : i32} : (tensor<4xi1>) -> tensor<i32>
    %1 = tosa.identity %0 : (tensor<i32>) -> tensor<i32>
    %2 = tosa.bitwise_not %1 : (tensor<i32>) -> tensor<i32>
    %3 = tosa.bitwise_xor %2, %0 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %3 : tensor<i32>
  }
}
