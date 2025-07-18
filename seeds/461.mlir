module {
  func.func @main(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<95x62x65x31xi32>, %arg3: tensor<1x1x1x31xi32>) -> (tensor<i32>, tensor<95x62x65x31xi32>) {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %1 = tosa.bitwise_xor %0, %0 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = tosa.minimum %arg2, %arg3 : (tensor<95x62x65x31xi32>, tensor<1x1x1x31xi32>) -> tensor<95x62x65x31xi32>
    %3 = tosa.minimum %2, %2 : (tensor<95x62x65x31xi32>, tensor<95x62x65x31xi32>) -> tensor<95x62x65x31xi32>
    return %1, %3 : tensor<i32>, tensor<95x62x65x31xi32>
  }
}
