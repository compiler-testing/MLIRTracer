module {
  func.func @main(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<62xi16>, %arg3: tensor<41x64x61xf32>) -> (tensor<62xi16>, tensor<1x1x1x1xi1>, tensor<41x64x61xf32>) {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1 = tosa.reverse %arg2 {axis = 0 : i32} : (tensor<62xi16>) -> tensor<62xi16>
    %2 = tosa.bitwise_not %1 : (tensor<62xi16>) -> tensor<62xi16>
    %r_3 = tosa.const_shape {values = dense<[ 1, 1, 1, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %3 = tosa.reshape %0, %r_3 : (tensor<i1>, !tosa.shape<4>) -> tensor<1x1x1x1xi1>
    %4 = tosa.bitwise_and %2, %1 : (tensor<62xi16>, tensor<62xi16>) -> tensor<62xi16>
    %5 = tosa.logical_xor %3, %3 : (tensor<1x1x1x1xi1>, tensor<1x1x1x1xi1>) -> tensor<1x1x1x1xi1>
    %6 = tosa.logical_not %5 : (tensor<1x1x1x1xi1>) -> tensor<1x1x1x1xi1>
    %7 = tosa.exp %arg3 : (tensor<41x64x61xf32>) -> tensor<41x64x61xf32>
    return %4, %6, %7 : tensor<62xi16>, tensor<1x1x1x1xi1>, tensor<41x64x61xf32>
  }
}
