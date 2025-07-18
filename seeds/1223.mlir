module {
  func.func @main(%arg0: tensor<77x56x50xi8>, %arg1: tensor<77x1x1xi8>, %arg2: tensor<8x87xi8>, %arg3: tensor<1x1xi8>, %arg4: tensor<41x64x97x2x44xf32>) -> (tensor<215600xi1>, tensor<41x64x97x2x44xf32>, tensor<8x87xi1>, tensor<41x64x97x2x44xf32>, tensor<8x87xi1>, tensor<1x87xi1>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<77x56x50xi8>, tensor<77x1x1xi8>) -> tensor<77x56x50xi1>
    %1 = tosa.greater %arg2, %arg3 : (tensor<8x87xi8>, tensor<1x1xi8>) -> tensor<8x87xi1>
    %2 = tosa.abs %0 : (tensor<77x56x50xi1>) -> tensor<77x56x50xi1>
    %3 = tosa.ceil %arg4 : (tensor<41x64x97x2x44xf32>) -> tensor<41x64x97x2x44xf32>
    %r_4 = tosa.const_shape {values = dense<[ 215600 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %4 = tosa.reshape %2, %r_4 : (tensor<77x56x50xi1>, !tosa.shape<1>) -> tensor<215600xi1>
    %5 = tosa.pow %3, %3 : (tensor<41x64x97x2x44xf32>, tensor<41x64x97x2x44xf32>) -> tensor<41x64x97x2x44xf32>
    %6 = tosa.logical_right_shift %1, %1 : (tensor<8x87xi1>, tensor<8x87xi1>) -> tensor<8x87xi1>
    %7 = tosa.logical_not %6 : (tensor<8x87xi1>) -> tensor<8x87xi1>
    %8 = tosa.logical_left_shift %7, %1 : (tensor<8x87xi1>, tensor<8x87xi1>) -> tensor<8x87xi1>
    %9 = tosa.maximum %3, %3 : (tensor<41x64x97x2x44xf32>, tensor<41x64x97x2x44xf32>) -> tensor<41x64x97x2x44xf32>
    %10 = tosa.bitwise_not %1 : (tensor<8x87xi1>) -> tensor<8x87xi1>
    %11 = tosa.reduce_all %1 {axis = 0 : i32} : (tensor<8x87xi1>) -> tensor<1x87xi1>
    %12 = tosa.bitwise_xor %11, %11 : (tensor<1x87xi1>, tensor<1x87xi1>) -> tensor<1x87xi1>
    %13 = tosa.logical_or %12, %11 : (tensor<1x87xi1>, tensor<1x87xi1>) -> tensor<1x87xi1>
    return %4, %5, %8, %9, %10, %13 : tensor<215600xi1>, tensor<41x64x97x2x44xf32>, tensor<8x87xi1>, tensor<41x64x97x2x44xf32>, tensor<8x87xi1>, tensor<1x87xi1>
  }
}
