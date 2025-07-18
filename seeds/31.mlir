module {
  func.func @main(%arg0: tensor<4x7x92xi8>, %arg1: tensor<4x92x63xi8>) -> tensor<4x7x63xi8> {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<4x7x92xi8>, tensor<4x92x63xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<4x7x63xi8>
    return %0 : tensor<4x7x63xi8>
  }
}
