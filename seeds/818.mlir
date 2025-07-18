module {
  func.func @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %1 = tosa.logical_right_shift %0, %0 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %1 : tensor<i32>
  }
}
