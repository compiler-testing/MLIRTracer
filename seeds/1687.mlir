module {
  func.func @main(%arg0: tensor<64x43xi16>, %arg1: tensor<64x1xi16>) -> tensor<64x43xi16> {
    %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<64x43xi16>, tensor<64x1xi16>) -> tensor<64x43xi16>
    return %0 : tensor<64x43xi16>
  }
}
