module {
  func.func @main(%arg0: tensor<100xi16>, %arg1: tensor<69x22xi64>) -> (tensor<i32>, tensor<22xi32>) {
    %0 = tosa.argmax %arg0 {axis = 0 : i32} : (tensor<100xi16>) -> tensor<i32>
    %1 = tosa.argmax %arg1 {axis = 0 : i32} : (tensor<69x22xi64>) -> tensor<22xi32>
    return %0, %1 : tensor<i32>, tensor<22xi32>
  }
}
