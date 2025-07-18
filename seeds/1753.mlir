module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<i1>, tensor<i32>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %1 = tosa.intdiv %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %0, %1 : tensor<i1>, tensor<i32>
  }
}
