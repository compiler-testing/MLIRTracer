module {
  func.func @main(%arg0: tensor<19xi1>, %arg1: tensor<19xi1>, %arg2: tensor<1x97x82x28x29xf32>) -> (tensor<1x97x82x28x29xf32>, tensor<19x1xi1>) {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<19xi1>, tensor<19xi1>) -> tensor<19xi1>
    %1 = tosa.clz %0 : (tensor<19xi1>) -> tensor<19xi1>
    %r_2 = tosa.const_shape {values = dense<[ 19, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %2 = tosa.reshape %1, %r_2 : (tensor<19xi1>, !tosa.shape<2>) -> tensor<19x1xi1>
    %3 = tosa.exp %arg2 : (tensor<1x97x82x28x29xf32>) -> tensor<1x97x82x28x29xf32>
    %4 = tosa.logical_not %2 : (tensor<19x1xi1>) -> tensor<19x1xi1>
    %5 = tosa.reverse %4 {axis = 1 : i32} : (tensor<19x1xi1>) -> tensor<19x1xi1>
    %6 = tosa.bitwise_not %5 : (tensor<19x1xi1>) -> tensor<19x1xi1>
    return %3, %6 : tensor<1x97x82x28x29xf32>, tensor<19x1xi1>
  }
}
