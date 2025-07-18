module {
  func.func @main(%arg0: tensor<19x27xf32>) -> (tensor<9x1x19x3xf32>, tensor<513xi1>) {
    %0 = tosa.reciprocal %arg0 : (tensor<19x27xf32>) -> tensor<19x27xf32>
    %r_1 = tosa.const_shape {values = dense<[ 9, 1, 19, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %1 = tosa.reshape %0, %r_1 : (tensor<19x27xf32>, !tosa.shape<4>) -> tensor<9x1x19x3xf32>
    %r_2 = tosa.const_shape {values = dense<[ 513 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %2 = tosa.reshape %1, %r_2 : (tensor<9x1x19x3xf32>, !tosa.shape<1>) -> tensor<513xf32>
    %3 = tosa.tanh %2 : (tensor<513xf32>) -> tensor<513xf32>
    %4 = tosa.equal %3, %3 : (tensor<513xf32>, tensor<513xf32>) -> tensor<513xi1>
    %5 = tosa.pow %1, %1 : (tensor<9x1x19x3xf32>, tensor<9x1x19x3xf32>) -> tensor<9x1x19x3xf32>
    %6 = tosa.arithmetic_right_shift %4, %4 {round = true} : (tensor<513xi1>, tensor<513xi1>) -> tensor<513xi1>
    %7 = tosa.logical_and %6, %6 : (tensor<513xi1>, tensor<513xi1>) -> tensor<513xi1>
    return %5, %7 : tensor<9x1x19x3xf32>, tensor<513xi1>
  }
}
