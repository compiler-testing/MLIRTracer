module {
  func.func @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i1> {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}
