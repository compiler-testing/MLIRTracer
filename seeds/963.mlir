module {
  func.func @main(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<87x71xi32>) -> (tensor<i1>, tensor<87x71xi32>) {
    %0 = tosa.equal %arg0, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %1 = tosa.reverse %arg2 {axis = 0 : i32} : (tensor<87x71xi32>) -> tensor<87x71xi32>
    return %0, %1 : tensor<i1>, tensor<87x71xi32>
  }
}
