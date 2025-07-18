module {
  func.func @main(%arg0: tensor<31x2x54x37x60x64xi32>, %arg1: tensor<1x2x54x37x60x1xi32>) -> tensor<31x2x54x37x60x64xi32> {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<31x2x54x37x60x64xi32>, tensor<1x2x54x37x60x1xi32>) -> tensor<31x2x54x37x60x64xi32>
    %1 = tosa.arithmetic_right_shift %0, %0 {round = true} : (tensor<31x2x54x37x60x64xi32>, tensor<31x2x54x37x60x64xi32>) -> tensor<31x2x54x37x60x64xi32>
    return %1 : tensor<31x2x54x37x60x64xi32>
  }
}
