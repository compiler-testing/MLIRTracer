module {
  func.func @main(%arg0: tensor<7x13x56x60x78xi16>) -> tensor<7x13x56x60x78xi16> {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<7x13x56x60x78xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<7x13x56x60x78xi16>
    return %0 : tensor<7x13x56x60x78xi16>
  }
}
