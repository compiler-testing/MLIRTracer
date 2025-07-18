module {
  func.func @main(%arg0: tensor<58x13xi1>) -> tensor<58x1xi1> {
    %0 = tosa.reduce_all %arg0 {axis = 1 : i32} : (tensor<58x13xi1>) -> tensor<58x1xi1>
    %1 = tosa.reverse %0 {axis = 0 : i32} : (tensor<58x1xi1>) -> tensor<58x1xi1>
    return %1 : tensor<58x1xi1>
  }
}
