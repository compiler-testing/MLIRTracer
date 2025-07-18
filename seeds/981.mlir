module {
  func.func @main(%arg0: tensor<66xi64>) -> tensor<i32> {
    %0 = tosa.reduce_product %arg0 {axis = 0 : i32} : (tensor<66xi64>) -> tensor<1xi64>
    %1 = tosa.reduce_max %0 {axis = 0 : i32} : (tensor<1xi64>) -> tensor<1xi64>
    %2 = tosa.argmax %1 {axis = 0 : i32} : (tensor<1xi64>) -> tensor<i32>
    return %2 : tensor<i32>
  }
}
