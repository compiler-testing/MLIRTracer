module {
  func.func @main(%arg0: tensor<72x91x67x13x24xi16>) -> tensor<72x91x67x13x24xi16> {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<72x91x67x13x24xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<72x91x67x13x24xi16>
    %1 = tosa.clamp %0 {min_val = -40 : i16, max_val = 83 : i16} : (tensor<72x91x67x13x24xi16>) -> tensor<72x91x67x13x24xi16>
    return %1 : tensor<72x91x67x13x24xi16>
  }
}
