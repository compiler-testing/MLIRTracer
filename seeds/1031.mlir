module {
  func.func @main(%arg0: tensor<53x11xi32>, %arg1: tensor<53x11xi32>) -> tensor<53x11xi1> {
    %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<53x11xi32>, tensor<53x11xi32>) -> tensor<53x11xi32>
    %1 = tosa.greater %0, %0 : (tensor<53x11xi32>, tensor<53x11xi32>) -> tensor<53x11xi1>
    %2 = tosa.bitwise_xor %1, %1 : (tensor<53x11xi1>, tensor<53x11xi1>) -> tensor<53x11xi1>
    return %2 : tensor<53x11xi1>
  }
}
