module {
  func.func @main(%arg0: tensor<60x65x30xi1>, %arg1: tensor<60x1x1xi1>, %arg2: tensor<8x3xi64>, %arg3: tensor<1x3xi64>, %arg4: tensor<90x65x92x19x4x57xf32>) -> (tensor<60x65x30xi1>, tensor<8x3xi1>, tensor<90x65x92x19x4x57xf32>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<60x65x30xi1>, tensor<60x1x1xi1>) -> tensor<60x65x30xi1>
    %1 = tosa.greater_equal %arg2, %arg3 : (tensor<8x3xi64>, tensor<1x3xi64>) -> tensor<8x3xi1>
    %2 = tosa.bitwise_not %1 : (tensor<8x3xi1>) -> tensor<8x3xi1>
    %3 = tosa.rsqrt %arg4 : (tensor<90x65x92x19x4x57xf32>) -> tensor<90x65x92x19x4x57xf32>
    %4 = tosa.rsqrt %3 : (tensor<90x65x92x19x4x57xf32>) -> tensor<90x65x92x19x4x57xf32>
    return %0, %2, %4 : tensor<60x65x30xi1>, tensor<8x3xi1>, tensor<90x65x92x19x4x57xf32>
  }
}
