module {
  func.func @main(%arg0: tensor<71x91x77x18xf32>, %arg1: tensor<71x1x1x1xf32>, %arg2: tensor<79x30x86x23xi16>, %arg3: tensor<1x30x86x1xi16>) -> (tensor<71x91x1x18xf32>, tensor<79x30x172x69xi16>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<71x91x77x18xf32>, tensor<71x1x1x1xf32>) -> tensor<71x91x77x18xf32>
    %1 = tosa.arithmetic_right_shift %arg2, %arg3 {round = true} : (tensor<79x30x86x23xi16>, tensor<1x30x86x1xi16>) -> tensor<79x30x86x23xi16>
    %2 = tosa.reduce_product %0 {axis = 2 : i32} : (tensor<71x91x77x18xf32>) -> tensor<71x91x1x18xf32>
    %3 = tosa.abs %2 : (tensor<71x91x1x18xf32>) -> tensor<71x91x1x18xf32>
    %4 = tosa.bitwise_xor %1, %1 : (tensor<79x30x86x23xi16>, tensor<79x30x86x23xi16>) -> tensor<79x30x86x23xi16>
    %t_5 = tosa.const_shape {values = dense<[ 1, 1, 2, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %5 = tosa.tile %4, %t_5 : (tensor<79x30x86x23xi16>, !tosa.shape<4>) -> tensor<79x30x172x69xi16>
    return %3, %5 : tensor<71x91x1x18xf32>, tensor<79x30x172x69xi16>
  }
}
