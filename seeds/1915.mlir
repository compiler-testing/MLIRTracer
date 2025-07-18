module {
  func.func @main(%arg0: tensor<9x11x37x16x43xi8>, %arg1: tensor<9x1x37x16x1xi8>) -> tensor<9x11x37x16x43xi8> {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<9x11x37x16x43xi8>, tensor<9x1x37x16x1xi8>) -> tensor<9x11x37x16x43xi8>
    return %0 : tensor<9x11x37x16x43xi8>
  }
}
