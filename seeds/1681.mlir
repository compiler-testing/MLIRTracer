module {
  func.func @main(%arg0: tensor<57x26x82xi32>, %arg1: tensor<57x82x59xi32>) -> tensor<57x26x59xi32> {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<57x26x82xi32>, tensor<57x82x59xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<57x26x59xi32>
    return %0 : tensor<57x26x59xi32>
  }
}
