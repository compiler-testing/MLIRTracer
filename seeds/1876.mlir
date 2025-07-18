module {
  func.func @main(%arg0: tensor<65x29x45x15xi8>) -> tensor<65x1x45xi32> {
    %0 = tosa.reduce_max %arg0 {axis = 1 : i32} : (tensor<65x29x45x15xi8>) -> tensor<65x1x45x15xi8>
    %1 = tosa.argmax %0 {axis = 3 : i32} : (tensor<65x1x45x15xi8>) -> tensor<65x1x45xi32>
    return %1 : tensor<65x1x45xi32>
  }
}
