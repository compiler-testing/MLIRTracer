module {
  func.func @main(%arg0: tensor<5x21xi64>) -> tensor<1x1xi64> {
    %0 = tosa.reduce_max %arg0 {axis = 1 : i32} : (tensor<5x21xi64>) -> tensor<5x1xi64>
    %1 = tosa.reduce_max %0 {axis = 0 : i32} : (tensor<5x1xi64>) -> tensor<1x1xi64>
    return %1 : tensor<1x1xi64>
  }
}
