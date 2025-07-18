module {
  func.func @main(%arg0: tensor<52x98x59xi32>, %arg1: tensor<52x59x15xi32>) -> tensor<52x98x15xi32> {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<52x98x59xi32>, tensor<52x59x15xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<52x98x15xi32>
    return %0 : tensor<52x98x15xi32>
  }
}
