module {
  func.func @main(%arg0: tensor<19x68xi64>, %arg1: tensor<19x1xi64>) -> tensor<1x1xi64> {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<19x68xi64>, tensor<19x1xi64>) -> tensor<19x68xi64>
    %1 = tosa.reduce_max %0 {axis = 1 : i32} : (tensor<19x68xi64>) -> tensor<19x1xi64>
    %2 = tosa.reduce_min %1 {axis = 0 : i32} : (tensor<19x1xi64>) -> tensor<1x1xi64>
    %3 = tosa.abs %2 : (tensor<1x1xi64>) -> tensor<1x1xi64>
    %4 = tosa.bitwise_not %3 : (tensor<1x1xi64>) -> tensor<1x1xi64>
    return %4 : tensor<1x1xi64>
  }
}
