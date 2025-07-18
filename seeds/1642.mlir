module {
  func.func @main(%arg0: tensor<98x97x25xi16>, %arg1: tensor<98x25x40xi16>) -> tensor<98x97x40xi16> {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<98x97x25xi16>, tensor<98x25x40xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<98x97x40xi16>
    return %0 : tensor<98x97x40xi16>
  }
}
