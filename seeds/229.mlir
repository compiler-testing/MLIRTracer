module {
  func.func @main(%arg0: tensor<93x32x37x39x39xi16>, %arg1: tensor<93x55x37x39x39xi16>) -> tensor<93x87x37x39x39xi16> {
    %0 = tosa.concat %arg0, %arg1 {axis = 1 : i32} : (tensor<93x32x37x39x39xi16>, tensor<93x55x37x39x39xi16>) -> tensor<93x87x37x39x39xi16>
    return %0 : tensor<93x87x37x39x39xi16>
  }
}
