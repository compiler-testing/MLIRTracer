module {
  func.func @main(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<25xi1>) -> (tensor<i1>, tensor<25xi1>) {
    %0 = tosa.equal %arg0, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1 = tosa.reverse %arg2 {axis = 0 : i32} : (tensor<25xi1>) -> tensor<25xi1>
    return %0, %1 : tensor<i1>, tensor<25xi1>
  }
}
