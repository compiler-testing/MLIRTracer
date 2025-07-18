module {
  func.func @main(%arg0: tensor<90x34xi8>, %arg1: tensor<1x34xi8>, %arg2: tensor<60x94x57xi1>, %arg3: tensor<60x94x57xi1>, %arg4: tensor<19x71x33x21xf32>) -> (tensor<90x34xi8>, tensor<19x71x33x21xf32>, tensor<60x94x57xi1>, tensor<60x94x57xi1>) {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = false} : (tensor<90x34xi8>, tensor<1x34xi8>) -> tensor<90x34xi8>
    %1 = tosa.logical_left_shift %0, %0 : (tensor<90x34xi8>, tensor<90x34xi8>) -> tensor<90x34xi8>
    %2 = tosa.bitwise_not %1 : (tensor<90x34xi8>) -> tensor<90x34xi8>
    %3 = tosa.logical_xor %arg2, %arg3 : (tensor<60x94x57xi1>, tensor<60x94x57xi1>) -> tensor<60x94x57xi1>
    %4 = tosa.clamp %2 {min_val = -46 : i8, max_val = 42 : i8} : (tensor<90x34xi8>) -> tensor<90x34xi8>
    %5 = tosa.rsqrt %arg4 : (tensor<19x71x33x21xf32>) -> tensor<19x71x33x21xf32>
    %6 = tosa.sigmoid %5 : (tensor<19x71x33x21xf32>) -> tensor<19x71x33x21xf32>
    %7 = tosa.logical_and %3, %3 : (tensor<60x94x57xi1>, tensor<60x94x57xi1>) -> tensor<60x94x57xi1>
    %in_zp_8 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_8 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %8 = tosa.negate %3, %in_zp_8, %out_zp_8 : (tensor<60x94x57xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<60x94x57xi1>
    return %4, %6, %7, %8 : tensor<90x34xi8>, tensor<19x71x33x21xf32>, tensor<60x94x57xi1>, tensor<60x94x57xi1>
  }
}
