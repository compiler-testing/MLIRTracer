module {
  func.func @main(%arg0: tensor<32x65x6x80x41x60xi1>, %arg1: tensor<1x1x1x80x41x1xi1>) -> tensor<32x65x6x80x41x60xi1> {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<32x65x6x80x41x60xi1>, tensor<1x1x1x80x41x1xi1>) -> tensor<32x65x6x80x41x60xi1>
    return %0 : tensor<32x65x6x80x41x60xi1>
  }
}
