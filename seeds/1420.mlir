module {
  func.func @main(%arg0: tensor<17x26x59x12x11xi1>, %arg1: tensor<17x1x59x12x11xi1>, %arg2: tensor<83x65x89xf32>) -> (tensor<17x26x59x12x11xi1>, tensor<83x65x89xi1>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<17x26x59x12x11xi1>, tensor<17x1x59x12x11xi1>) -> tensor<17x26x59x12x11xi1>
    %1 = tosa.logical_and %0, %0 : (tensor<17x26x59x12x11xi1>, tensor<17x26x59x12x11xi1>) -> tensor<17x26x59x12x11xi1>
    %2 = tosa.floor %arg2 : (tensor<83x65x89xf32>) -> tensor<83x65x89xf32>
    %3 = tosa.clamp %2 {min_val = 4.000000e+00 : f32, max_val = 1.500000e+02 : f32} : (tensor<83x65x89xf32>) -> tensor<83x65x89xf32>
    %4 = tosa.bitwise_or %1, %1 : (tensor<17x26x59x12x11xi1>, tensor<17x26x59x12x11xi1>) -> tensor<17x26x59x12x11xi1>
    %5 = tosa.minimum %3, %3 : (tensor<83x65x89xf32>, tensor<83x65x89xf32>) -> tensor<83x65x89xf32>
    %6 = tosa.equal %3, %5 : (tensor<83x65x89xf32>, tensor<83x65x89xf32>) -> tensor<83x65x89xi1>
    %7 = tosa.logical_and %6, %6 : (tensor<83x65x89xi1>, tensor<83x65x89xi1>) -> tensor<83x65x89xi1>
    %8 = tosa.logical_right_shift %7, %6 : (tensor<83x65x89xi1>, tensor<83x65x89xi1>) -> tensor<83x65x89xi1>
    return %4, %8 : tensor<17x26x59x12x11xi1>, tensor<83x65x89xi1>
  }
}
