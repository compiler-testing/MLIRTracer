module {
  func.func @main(%arg0: tensor<82x55xi1>) -> tensor<82x1xi1> {
    %0 = tosa.reduce_all %arg0 {axis = 1 : i32} : (tensor<82x55xi1>) -> tensor<82x1xi1>
    %1 = tosa.reverse %0 {axis = 1 : i32} : (tensor<82x1xi1>) -> tensor<82x1xi1>
    %2 = tosa.clz %1 : (tensor<82x1xi1>) -> tensor<82x1xi1>
    return %2 : tensor<82x1xi1>
  }
}
