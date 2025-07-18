module {
  func.func @main(%arg0: tensor<47x66x10xi16>, %arg1: tensor<47x1x10xi16>) -> tensor<47x1x10xi16> {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<47x66x10xi16>, tensor<47x1x10xi16>) -> tensor<47x66x10xi16>
    %1 = tosa.reduce_min %0 {axis = 1 : i32} : (tensor<47x66x10xi16>) -> tensor<47x1x10xi16>
    return %1 : tensor<47x1x10xi16>
  }
}
