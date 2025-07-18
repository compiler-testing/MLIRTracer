module {
  func.func @main(%arg0: tensor<68x70x90x64xi16>) -> tensor<68x70x90x1xi16> {
    %0 = tosa.reduce_max %arg0 {axis = 3 : i32} : (tensor<68x70x90x64xi16>) -> tensor<68x70x90x1xi16>
    return %0 : tensor<68x70x90x1xi16>
  }
}
