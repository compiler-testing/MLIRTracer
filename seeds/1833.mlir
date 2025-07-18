module {
  func.func @main(%arg0: tensor<36x7x9xi1>) -> tensor<1x7x1xi1> {
    %0 = tosa.logical_not %arg0 : (tensor<36x7x9xi1>) -> tensor<36x7x9xi1>
    %1 = tosa.reduce_max %0 {axis = 0 : i32} : (tensor<36x7x9xi1>) -> tensor<1x7x9xi1>
    %2 = tosa.reverse %1 {axis = 0 : i32} : (tensor<1x7x9xi1>) -> tensor<1x7x9xi1>
    %3 = tosa.logical_not %2 : (tensor<1x7x9xi1>) -> tensor<1x7x9xi1>
    %4 = tosa.sub %3, %2 : (tensor<1x7x9xi1>, tensor<1x7x9xi1>) -> tensor<1x7x9xi1>
    %5 = tosa.reduce_max %4 {axis = 2 : i32} : (tensor<1x7x9xi1>) -> tensor<1x7x1xi1>
    return %5 : tensor<1x7x1xi1>
  }
}
