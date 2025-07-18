module {
  func.func @main(%arg0: tensor<2xi16>, %arg1: tensor<9xi16>) -> tensor<11xi16> {
    %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<2xi16>, tensor<9xi16>) -> tensor<11xi16>
    return %0 : tensor<11xi16>
  }
}
