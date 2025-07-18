module {
  func.func @main(%arg0: tensor<57x54x10x47xi1>, %arg1: tensor<1x1x10x1xi1>) -> tensor<57x54x10x47xi1> {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<57x54x10x47xi1>, tensor<1x1x10x1xi1>) -> tensor<57x54x10x47xi1>
    return %0 : tensor<57x54x10x47xi1>
  }
}
