module {
  func.func @main(%arg0: tensor<94x45x23xi8>, %arg1: tensor<94x23x74xi8>) -> tensor<94x45x74xi8> {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<94x45x23xi8>, tensor<94x23x74xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<94x45x74xi8>
    return %0 : tensor<94x45x74xi8>
  }
}
