module {
  func.func @main(%arg0: tensor<18x12x79xi1>, %arg1: tensor<1x1x79xi1>, %arg2: tensor<52x76xf32>) -> (tensor<18x12x79xi1>, tensor<52x76xf32>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<18x12x79xi1>, tensor<1x1x79xi1>) -> tensor<18x12x79xi1>
    %1 = tosa.logical_right_shift %0, %0 : (tensor<18x12x79xi1>, tensor<18x12x79xi1>) -> tensor<18x12x79xi1>
    %2 = tosa.rsqrt %arg2 : (tensor<52x76xf32>) -> tensor<52x76xf32>
    %in_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %3 = tosa.negate %1, %in_zp_3, %out_zp_3 : (tensor<18x12x79xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<18x12x79xi1>
    %4 = tosa.bitwise_xor %3, %3 : (tensor<18x12x79xi1>, tensor<18x12x79xi1>) -> tensor<18x12x79xi1>
    %in_zp_5 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_5 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5 = tosa.negate %2, %in_zp_5, %out_zp_5 : (tensor<52x76xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<52x76xf32>
    return %4, %5 : tensor<18x12x79xi1>, tensor<52x76xf32>
  }
}
