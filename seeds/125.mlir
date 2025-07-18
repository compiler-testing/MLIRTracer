module {
  func.func @main(%arg0: tensor<95x78x17x40x60x51xi64>, %arg1: tensor<1x78x1x1x60x51xi64>) -> tensor<95x78x17x40x60x51xi64> {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<95x78x17x40x60x51xi64>, tensor<1x78x1x1x60x51xi64>) -> tensor<95x78x17x40x60x51xi64>
    return %0 : tensor<95x78x17x40x60x51xi64>
  }
}
