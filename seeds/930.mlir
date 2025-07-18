module {
  func.func @main(%arg0: tensor<22x52x55xi16>, %arg1: tensor<32x69x50xf32>, %arg2: tensor<77x56xi1>) -> (tensor<32x69x50xf32>, tensor<22x1x55xi16>, tensor<1x56xi1>, tensor<32x69x50xf32>) {
    %0 = tosa.reduce_min %arg0 {axis = 1 : i32} : (tensor<22x52x55xi16>) -> tensor<22x1x55xi16>
    %1 = tosa.floor %arg1 : (tensor<32x69x50xf32>) -> tensor<32x69x50xf32>
    %2 = tosa.bitwise_and %0, %0 : (tensor<22x1x55xi16>, tensor<22x1x55xi16>) -> tensor<22x1x55xi16>
    %in_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.negate %1, %in_zp_3, %out_zp_3 : (tensor<32x69x50xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<32x69x50xf32>
    %4 = tosa.reduce_any %arg2 {axis = 0 : i32} : (tensor<77x56xi1>) -> tensor<1x56xi1>
    %in_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %out_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %5 = tosa.negate %2, %in_zp_5, %out_zp_5 : (tensor<22x1x55xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<22x1x55xi16>
    %6 = tosa.logical_right_shift %4, %4 : (tensor<1x56xi1>, tensor<1x56xi1>) -> tensor<1x56xi1>
    %7 = tosa.reciprocal %1 : (tensor<32x69x50xf32>) -> tensor<32x69x50xf32>
    return %3, %5, %6, %7 : tensor<32x69x50xf32>, tensor<22x1x55xi16>, tensor<1x56xi1>, tensor<32x69x50xf32>
  }
}
