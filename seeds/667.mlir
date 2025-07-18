module {
  func.func @main(%arg0: tensor<53xi64>, %arg1: tensor<1xi64>, %arg2: tensor<90x31x12x85xf32>, %arg3: tensor<1x1x1x1xf32>) -> (tensor<1xi1>, tensor<90x31x12x85xf32>) {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = false} : (tensor<53xi64>, tensor<1xi64>) -> tensor<53xi64>
    %1 = tosa.greater_equal %0, %0 : (tensor<53xi64>, tensor<53xi64>) -> tensor<53xi1>
    %2 = tosa.reduce_max %1 {axis = 0 : i32} : (tensor<53xi1>) -> tensor<1xi1>
    %3 = tosa.sub %2, %2 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %4 = tosa.pow %arg2, %arg3 : (tensor<90x31x12x85xf32>, tensor<1x1x1x1xf32>) -> tensor<90x31x12x85xf32>
    return %3, %4 : tensor<1xi1>, tensor<90x31x12x85xf32>
  }
}
