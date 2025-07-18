module {
  func.func @main(%arg0: tensor<23xi1>, %arg1: tensor<1xi1>) -> tensor<23xi1> {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<23xi1>, tensor<1xi1>) -> tensor<23xi1>
    return %0 : tensor<23xi1>
  }
}
