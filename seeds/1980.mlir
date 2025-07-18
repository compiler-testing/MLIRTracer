module {
  func.func @main(%arg0: tensor<63x26xi16>, %arg1: tensor<63x99xi16>) -> tensor<63x125xi16> {
    %0 = tosa.concat %arg0, %arg1 {axis = 1 : i32} : (tensor<63x26xi16>, tensor<63x99xi16>) -> tensor<63x125xi16>
    return %0 : tensor<63x125xi16>
  }
}
