module {
  func.func @main(%arg0: tensor<75x53xi1>, %arg1: tensor<75x53xi1>) -> tensor<75x1xi1> {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<75x53xi1>, tensor<75x53xi1>) -> tensor<75x53xi1>
    %1 = tosa.reduce_max %0 {axis = 1 : i32} : (tensor<75x53xi1>) -> tensor<75x1xi1>
    return %1 : tensor<75x1xi1>
  }
}
