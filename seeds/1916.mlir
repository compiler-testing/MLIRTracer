module {
  func.func @main(%arg0: tensor<20xi1>) -> (tensor<i32>, tensor<1xi1>) {
    %0 = tosa.reduce_any %arg0 {axis = 0 : i32} : (tensor<20xi1>) -> tensor<1xi1>
    %1 = tosa.argmax %0 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<i32>
    %2 = tosa.logical_or %0, %0 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    return %1, %2 : tensor<i32>, tensor<1xi1>
  }
}
