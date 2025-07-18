module {
  func.func @main(%arg0: tensor<69x22xi64>, %arg1: tensor<65xi1>) -> (tensor<1xi1>, tensor<69x1xi1>) {
    %0 = tosa.reduce_sum %arg0 {axis = 1 : i32} : (tensor<69x22xi64>) -> tensor<69x1xi64>
    %1 = tosa.reduce_any %arg1 {axis = 0 : i32} : (tensor<65xi1>) -> tensor<1xi1>
    %2 = tosa.identity %0 : (tensor<69x1xi64>) -> tensor<69x1xi64>
    %3 = tosa.reduce_max %1 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %4 = tosa.abs %3 : (tensor<1xi1>) -> tensor<1xi1>
    %5 = tosa.bitwise_or %4, %4 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %6 = tosa.bitwise_or %2, %0 : (tensor<69x1xi64>, tensor<69x1xi64>) -> tensor<69x1xi64>
    %7 = tosa.greater_equal %6, %2 : (tensor<69x1xi64>, tensor<69x1xi64>) -> tensor<69x1xi1>
    return %5, %7 : tensor<1xi1>, tensor<69x1xi1>
  }
}
