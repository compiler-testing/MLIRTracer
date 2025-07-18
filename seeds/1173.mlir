module {
  func.func @main(%arg0: tensor<51x5x33x97x93xi1>, %arg1: tensor<51x1x1x97x1xi1>, %arg2: tensor<81x39xf32>, %arg3: tensor<1x1xf32>) -> (tensor<81x39xf32>, tensor<51x5x33x97x93xi1>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<51x5x33x97x93xi1>, tensor<51x1x1x97x1xi1>) -> tensor<51x5x33x97x93xi1>
    %1 = tosa.logical_not %0 : (tensor<51x5x33x97x93xi1>) -> tensor<51x5x33x97x93xi1>
    %2 = tosa.logical_and %1, %1 : (tensor<51x5x33x97x93xi1>, tensor<51x5x33x97x93xi1>) -> tensor<51x5x33x97x93xi1>
    %3 = tosa.logical_left_shift %2, %2 : (tensor<51x5x33x97x93xi1>, tensor<51x5x33x97x93xi1>) -> tensor<51x5x33x97x93xi1>
    %in_zp_4 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_4 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %4 = tosa.negate %3, %in_zp_4, %out_zp_4 : (tensor<51x5x33x97x93xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<51x5x33x97x93xi1>
    %5 = tosa.minimum %arg2, %arg3 : (tensor<81x39xf32>, tensor<1x1xf32>) -> tensor<81x39xf32>
    %6 = tosa.bitwise_and %4, %1 : (tensor<51x5x33x97x93xi1>, tensor<51x5x33x97x93xi1>) -> tensor<51x5x33x97x93xi1>
    %7 = tosa.abs %6 : (tensor<51x5x33x97x93xi1>) -> tensor<51x5x33x97x93xi1>
    return %5, %7 : tensor<81x39xf32>, tensor<51x5x33x97x93xi1>
  }
}
