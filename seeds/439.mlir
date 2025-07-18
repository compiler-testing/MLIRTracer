module {
  func.func @main(%arg0: tensor<70x71xf32>, %arg1: tensor<63x6x60xi64>, %arg2: tensor<63x1x60xi64>) -> (tensor<70x71xf32>, tensor<63x6x60xi64>) {
    %0 = tosa.rsqrt %arg0 : (tensor<70x71xf32>) -> tensor<70x71xf32>
    %1 = tosa.bitwise_xor %arg1, %arg2 : (tensor<63x6x60xi64>, tensor<63x1x60xi64>) -> tensor<63x6x60xi64>
    %2 = tosa.logical_left_shift %1, %1 : (tensor<63x6x60xi64>, tensor<63x6x60xi64>) -> tensor<63x6x60xi64>
    return %0, %2 : tensor<70x71xf32>, tensor<63x6x60xi64>
  }
}
