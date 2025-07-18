module {
  func.func @main(%arg0: tensor<26x70xi32>) -> tensor<1x1xi1> {
    %0 = tosa.reduce_sum %arg0 {axis = 1 : i32} : (tensor<26x70xi32>) -> tensor<26x1xi32>
    %1 = tosa.bitwise_xor %0, %0 : (tensor<26x1xi32>, tensor<26x1xi32>) -> tensor<26x1xi32>
    %2 = tosa.minimum %1, %1 : (tensor<26x1xi32>, tensor<26x1xi32>) -> tensor<26x1xi32>
    %3 = tosa.greater %2, %1 : (tensor<26x1xi32>, tensor<26x1xi32>) -> tensor<26x1xi1>
    %4 = tosa.abs %3 : (tensor<26x1xi1>) -> tensor<26x1xi1>
    %in_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %5 = tosa.negate %4, %in_zp_5, %out_zp_5 : (tensor<26x1xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<26x1xi1>
    %6 = tosa.reduce_min %5 {axis = 0 : i32} : (tensor<26x1xi1>) -> tensor<1x1xi1>
    return %6 : tensor<1x1xi1>
  }
}
