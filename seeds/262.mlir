module {
  func.func @main(%arg0: tensor<100x98xi16>, %arg1: tensor<1x98xi16>) -> tensor<1x98xi16> {
    %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<100x98xi16>, tensor<1x98xi16>) -> tensor<100x98xi16>
    %1 = tosa.reduce_max %0 {axis = 0 : i32} : (tensor<100x98xi16>) -> tensor<1x98xi16>
    return %1 : tensor<1x98xi16>
  }
}
