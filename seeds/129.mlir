module {
  func.func @main(%arg0: tensor<9x57xi32>, %arg1: tensor<1x1xi32>, %arg2: tensor<16x93xf32>) -> (tensor<9x57xi32>, tensor<16x93xf32>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<9x57xi32>, tensor<1x1xi32>) -> tensor<9x57xi32>
    %1 = tosa.bitwise_not %0 : (tensor<9x57xi32>) -> tensor<9x57xi32>
    %2 = tosa.bitwise_not %1 : (tensor<9x57xi32>) -> tensor<9x57xi32>
    %3 = tosa.intdiv %2, %2 : (tensor<9x57xi32>, tensor<9x57xi32>) -> tensor<9x57xi32>
    %4 = tosa.logical_left_shift %3, %3 : (tensor<9x57xi32>, tensor<9x57xi32>) -> tensor<9x57xi32>
    %5 = tosa.rsqrt %arg2 : (tensor<16x93xf32>) -> tensor<16x93xf32>
    return %4, %5 : tensor<9x57xi32>, tensor<16x93xf32>
  }
}
