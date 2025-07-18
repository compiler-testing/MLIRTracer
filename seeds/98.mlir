module {
  func.func @main(%arg0: tensor<i32>, %arg1: tensor<98x31xf32>) -> (tensor<i1>, tensor<98x31xf32>) {
    %0 = tosa.abs %arg0 : (tensor<i32>) -> tensor<i32>
    %1 = tosa.rsqrt %arg1 : (tensor<98x31xf32>) -> tensor<98x31xf32>
    %2 = tosa.greater %0, %0 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %in_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %3 = tosa.negate %2, %in_zp_3, %out_zp_3 : (tensor<i1>, tensor<1xi1>, tensor<1xi1>) -> tensor<i1>
    %4 = tosa.log %1 : (tensor<98x31xf32>) -> tensor<98x31xf32>
    return %3, %4 : tensor<i1>, tensor<98x31xf32>
  }
}
