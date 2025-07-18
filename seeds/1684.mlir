module {
  func.func @main(%arg0: tensor<i32>) -> tensor<i32> {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<i32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
    return %0 : tensor<i32>
  }
}
