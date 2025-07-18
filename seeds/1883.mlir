module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<70x81xi1>) -> (tensor<70x1xi1>, tensor<i1>) {
    %0 = tosa.clz %arg0 : (tensor<i1>) -> tensor<i1>
    %1 = tosa.reduce_all %arg1 {axis = 1 : i32} : (tensor<70x81xi1>) -> tensor<70x1xi1>
    %2 = tosa.clz %0 : (tensor<i1>) -> tensor<i1>
    return %1, %2 : tensor<70x1xi1>, tensor<i1>
  }
}
