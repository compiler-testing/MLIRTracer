module {
  func.func @main(%arg0: tensor<5x62xi8>) -> tensor<5x1xi8> {
    %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<5x62xi8>) -> tensor<5x62xi8>
    %1 = tosa.reduce_max %0 {axis = 1 : i32} : (tensor<5x62xi8>) -> tensor<5x1xi8>
    return %1 : tensor<5x1xi8>
  }
}
