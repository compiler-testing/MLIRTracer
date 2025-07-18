module {
  func.func @main(%arg0: tensor<23x2xi16>, %arg1: tensor<52x89xf32>, %arg2: tensor<i1>) -> (tensor<23x2xi16>, tensor<i1>, tensor<52x89xf32>, tensor<52x89xi1>) {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<23x2xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<23x2xi16>
    %in_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %out_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %1 = tosa.negate %0, %in_zp_1, %out_zp_1 : (tensor<23x2xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<23x2xi16>
    %2 = tosa.logical_left_shift %1, %1 : (tensor<23x2xi16>, tensor<23x2xi16>) -> tensor<23x2xi16>
    %3 = tosa.ceil %arg1 : (tensor<52x89xf32>) -> tensor<52x89xf32>
    %4 = tosa.sub %3, %3 : (tensor<52x89xf32>, tensor<52x89xf32>) -> tensor<52x89xf32>
    %5 = tosa.bitwise_xor %2, %1 : (tensor<23x2xi16>, tensor<23x2xi16>) -> tensor<23x2xi16>
    %6 = tosa.bitwise_xor %5, %2 : (tensor<23x2xi16>, tensor<23x2xi16>) -> tensor<23x2xi16>
    %7 = tosa.logical_left_shift %6, %6 : (tensor<23x2xi16>, tensor<23x2xi16>) -> tensor<23x2xi16>
    %8 = tosa.logical_not %arg2 : (tensor<i1>) -> tensor<i1>
    %9 = tosa.ceil %4 : (tensor<52x89xf32>) -> tensor<52x89xf32>
    %10 = tosa.reciprocal %9 : (tensor<52x89xf32>) -> tensor<52x89xf32>
    %11 = tosa.bitwise_xor %8, %8 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %12 = tosa.ceil %4 : (tensor<52x89xf32>) -> tensor<52x89xf32>
    %13 = tosa.greater %4, %10 : (tensor<52x89xf32>, tensor<52x89xf32>) -> tensor<52x89xi1>
    return %7, %11, %12, %13 : tensor<23x2xi16>, tensor<i1>, tensor<52x89xf32>, tensor<52x89xi1>
  }
}
