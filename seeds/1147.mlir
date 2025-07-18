module {
  func.func @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = tosa.add %arg0, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %0 : tensor<i32>
  }
}
