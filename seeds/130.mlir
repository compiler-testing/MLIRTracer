module {
  func.func @main(%arg0: tensor<63x45xi16>) -> tensor<63x1xi16> {
    %0 = tosa.reduce_max %arg0 {axis = 1 : i32} : (tensor<63x45xi16>) -> tensor<63x1xi16>
    return %0 : tensor<63x1xi16>
  }
}
