module {
  func.func @main(%arg0: tensor<41x51x63xi16>, %arg1: tensor<1x1x1xi16>) -> tensor<41x51x63xi16> {
    %0 = tosa.add %arg0, %arg1 : (tensor<41x51x63xi16>, tensor<1x1x1xi16>) -> tensor<41x51x63xi16>
    return %0 : tensor<41x51x63xi16>
  }
}
