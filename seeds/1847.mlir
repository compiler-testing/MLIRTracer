module {
  func.func @main(%arg0: tensor<69xi16>, %arg1: tensor<14x90x62x74xi32>, %arg2: tensor<14x1x1x74xi32>) -> (tensor<69xi16>, tensor<14x90x62x74xi32>) {
    %0 = tosa.abs %arg0 : (tensor<69xi16>) -> tensor<69xi16>
    %1 = tosa.intdiv %arg1, %arg2 : (tensor<14x90x62x74xi32>, tensor<14x1x1x74xi32>) -> tensor<14x90x62x74xi32>
    return %0, %1 : tensor<69xi16>, tensor<14x90x62x74xi32>
  }
}
