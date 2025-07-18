module {
  func.func @main(%arg0: tensor<74x65xi16>) -> tensor<74x1xi16> {
    %0 = tosa.reduce_max %arg0 {axis = 1 : i32} : (tensor<74x65xi16>) -> tensor<74x1xi16>
    %1 = tosa.identity %0 : (tensor<74x1xi16>) -> tensor<74x1xi16>
    %2 = tosa.clz %1 : (tensor<74x1xi16>) -> tensor<74x1xi16>
    %3 = tosa.logical_left_shift %2, %2 : (tensor<74x1xi16>, tensor<74x1xi16>) -> tensor<74x1xi16>
    %4 = tosa.logical_left_shift %3, %2 : (tensor<74x1xi16>, tensor<74x1xi16>) -> tensor<74x1xi16>
    return %4 : tensor<74x1xi16>
  }
}
