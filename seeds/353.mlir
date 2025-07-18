module {
  func.func @main(%arg0: tensor<61x22x46xf32>, %arg1: tensor<61x46x76xf32>) -> tensor<61x22x76xf32> {
    %a_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %b_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<61x22x46xf32>, tensor<61x46x76xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<61x22x76xf32>
    return %0 : tensor<61x22x76xf32>
  }
}
