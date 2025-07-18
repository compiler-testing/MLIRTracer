module {
  func.func @main(%arg0: tensor<44xi64>) -> tensor<1xi64> {
    %0 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<44xi64>) -> tensor<1xi64>
    return %0 : tensor<1xi64>
  }
}
