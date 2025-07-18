module {
  func.func @main(%arg0: tensor<16x32xi16>, %arg1: tensor<1x1xi16>, %arg2: tensor<60x91x4xf32>) -> (tensor<5x1xi16>, tensor<60x91x4xf32>) {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = true} : (tensor<16x32xi16>, tensor<1x1xi16>) -> tensor<16x32xi16>
    %s_1_start = tosa.const_shape {values = dense<[ 7, 8 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_1_size = tosa.const_shape {values = dense<[ 5, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.slice %0, %s_1_start, %s_1_size : (tensor<16x32xi16>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<5x1xi16>
    %2 = tosa.reduce_min %1 {axis = 1 : i32} : (tensor<5x1xi16>) -> tensor<5x1xi16>
    %3 = tosa.add %2, %1 : (tensor<5x1xi16>, tensor<5x1xi16>) -> tensor<5x1xi16>
    %4 = tosa.floor %arg2 : (tensor<60x91x4xf32>) -> tensor<60x91x4xf32>
    %5 = tosa.bitwise_xor %3, %2 : (tensor<5x1xi16>, tensor<5x1xi16>) -> tensor<5x1xi16>
    %6 = tosa.sigmoid %4 : (tensor<60x91x4xf32>) -> tensor<60x91x4xf32>
    return %5, %6 : tensor<5x1xi16>, tensor<60x91x4xf32>
  }
}
